import torch
import numpy as np
from dqn_agent import DQNAgent
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

state_size = 4
action_size = 2  # 0 = RF, 1 = PLC
dqn_agent = DQNAgent(state_size, action_size)
dqn_agent.load("trained_models/dqn_model.pth")
dqn_agent.epsilon = 0 

dqn_agent.model.to(torch.device("cpu"))
dqn_agent.model.eval()

@torch.no_grad()
def select_comm_mode(signal_rf, latency_rf, signal_plc, latency_plc):
    state = np.array([signal_rf, latency_rf, signal_plc, latency_plc], dtype=np.float32)
    state_tensor = torch.FloatTensor(state).unsqueeze(0) 
    q_values = dqn_agent.model(state_tensor)
    action = torch.argmax(q_values, dim=1).item()
    return "RF" if action == 0 else "PLC"

if __name__ == "__main__":
    start = time.time()
    mode = select_comm_mode(-87.143159, 46.598387, -87.806131, 115.937354)
    end = time.time()
    print("Recommended Communication Mode:", mode)
    print(f"Inference Time: {end - start:.4f} seconds")
