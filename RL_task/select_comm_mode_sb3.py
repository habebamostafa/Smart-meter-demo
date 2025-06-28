import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env.custom_comm_env import CustomCommEnv
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

dummy_data = pd.DataFrame({
    'signal_rf': [0],
    'latency_rf': [0],
    'signal_plc': [0],
    'latency_plc': [0],
    'reward_rf': [0],
    'reward_plc': [0]
})
env = CustomCommEnv(dummy_data)

model = PPO.load("trained_models/ppo_comm_mode_sb3")

def select_comm_mode_ppo(signal_rf, latency_rf, signal_plc, latency_plc):
    obs = np.array([signal_rf, latency_rf, signal_plc, latency_plc], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return "RF" if action == 0 else "PLC"

if __name__ == "__main__":
    start = time.time()
    mode = select_comm_mode_ppo(-14.14, 46.6, -87.80, 15.9)
    # mode = select_comm_mode_ppo(-87.143159, 46.598387, -87.806131, 115.937354)
    end = time.time()
    print("Recommended Communication Mode (PPO SB3):", mode)
    print(f"Inference Time: {end - start:.4f} seconds")
