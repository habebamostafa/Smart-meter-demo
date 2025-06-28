import serial
import time
import json
import torch
import numpy as np
from tensorflow.keras.models import load_model
from dqn_agent import DQNAgent
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

SERIAL_PORT = 'COM5'
BAUD_RATE = 9600
TIMEOUT = 1  # seconds
IDLE_TIMEOUT_SECONDS = 60  # Exit if no data received in 60 seconds

dqn_agent = DQNAgent(state_size=4, action_size=2)
dqn_agent.load("trained_models/dqn_model.pth")
dqn_agent.epsilon = 0
dqn_agent.model.to(torch.device("cpu"))
dqn_agent.model.eval()

autoencoder = load_model("trained_models/autoencoder_model.h5", compile=False)

# --------- ANOMALY THRESHOLD ----------
ANOMALY_THRESHOLD = 1.5579  

@torch.no_grad()
def select_comm_mode(signal_rf, latency_rf, signal_plc, latency_plc):
    state = np.array([signal_rf, latency_rf, signal_plc, latency_plc], dtype=np.float32)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = dqn_agent.model(state_tensor)
    action = torch.argmax(q_values, dim=1).item()
    return "RF" if action == 0 else "PLC"

# def is_anomaly(signal_rf, latency_rf, signal_plc, latency_plc):
#     input_vec = np.array([[signal_rf, latency_rf, signal_plc, latency_plc]], dtype=np.float32)
#     reconstructed = autoencoder.predict(input_vec, verbose=0)
#     mse = np.mean((input_vec - reconstructed) ** 2)
#     return mse > ANOMALY_THRESHOLD

def is_anomaly(input_features):
    input_vec = np.array([input_features], dtype=np.float32)
    reconstructed = autoencoder.predict(input_vec, verbose=0)
    mse = np.mean((input_vec - reconstructed) ** 2)
    return (mse > ANOMALY_THRESHOLD), mse

def parse_line(line):
    try:
        # Split by comma, then strip whitespace from each part
        parts = [p.strip() for p in line.split(',')]

        # Extract relevant values
        signal_rf = float(parts[1])
        latency_rf = float(parts[2])
        signal_plc = float(parts[4])
        latency_plc = float(parts[5])
    
        extra1 = float(parts[8]) if len(parts) > 8 else 0.0
        extra2 = 0.0  
        extra3 = 0.0  

        return [signal_rf, latency_rf, signal_plc, latency_plc, extra1, extra2, extra3]
    except Exception as e:
        print("Parsing error:", e)
        return None

# def parse_line(line):
#     try:
#         # Split by comma, then strip whitespace from each part
#         parts = [p.strip() for p in line.split(',')]
        
#         # Extract relevant values
#         signal_rf = float(parts[1])               # -84.27411319
#         latency_rf = float(parts[2])              # 129.1229140198042
#         signal_plc = float(parts[4])              # -84.1197631
#         latency_plc = float(parts[5])             # 25.94008182537916

#         return signal_rf, latency_rf, signal_plc, latency_plc
#     except Exception as e:
#         print("Parsing error:", e)
#         return None

def write_structured_output(timestamp, rl_mode, anomaly_flag, mse_score):
    try:
        with open("model_outputs.json", "r") as f:
            current = json.load(f)
    except:
        current = {
            "forecast": [],
            "anomalies": [],
            "communication_modes": []
        }

    if anomaly_flag:
        current["anomalies"].append({
            "timestamp": timestamp,
            "severity": round(mse_score, 3)
        })

    current["communication_modes"].append({
        "timestamp": timestamp,
        "mode": rl_mode
    })

    with open("model_outputs.json", "w") as f:
        json.dump(current, f, indent=2)

    return current
import subprocess

subprocess.run(["python", "./cybersecurity.py", "model_outputs.json"])


def reply_to_microcontroller(rl_mode):
    return 'r' if rl_mode == "RF" else 'p'

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
time.sleep(2)
print("Listening on serial...")

last_data_time = time.time()

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print("Received:", line)

            data = parse_line(line)
            if data:
                last_data_time = time.time()
                signal_rf, latency_rf, signal_plc, latency_plc = data[:4]

                rl_output = select_comm_mode(signal_rf, latency_rf, signal_plc, latency_plc)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                anomaly_flag, mse_score = is_anomaly(data)
                result = write_structured_output(timestamp, rl_output, anomaly_flag, mse_score)
                print("Output:", result)

                decision = reply_to_microcontroller(rl_output)
                ser.write((decision + '\n').encode('utf-8'))
                print(f"Sent decision to microcontroller: {decision}")

        if time.time() - last_data_time > IDLE_TIMEOUT_SECONDS:
            print(f"No data received for {IDLE_TIMEOUT_SECONDS} seconds. Exiting...")
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    ser.close()
    print("Serial connection closed.")