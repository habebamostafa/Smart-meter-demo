import torch
import numpy as np
import time
from customed_ppo_agent import PPOAgent
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def normalize_state(signal_rf, latency_rf, signal_plc, latency_plc):
    signal_rf = (signal_rf + 120) / 60
    signal_plc = (signal_plc + 120) / 60
    latency_rf = latency_rf / 200
    latency_plc = latency_plc / 200
    return [signal_rf, latency_rf, signal_plc, latency_plc]

@torch.no_grad()
def select_comm_mode_ppo(signal_rf, latency_rf, signal_plc, latency_plc):
    state = normalize_state(signal_rf, latency_rf, signal_plc, latency_plc)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    agent = PPOAgent(state_dim=4, action_dim=2)
    agent.load("trained_models/customed_ppo_model.pth")

    probs, _ = agent.model(state_tensor)
    action = torch.argmax(probs, dim=1).item()
    return "RF" if action == 0 else "PLC"

if __name__ == "__main__":
    start = time.time()
    mode = select_comm_mode_ppo(-14.143159, 46.598387, -87.806131, 115.937354)
    end = time.time()
    print("Recommended Communication Mode (PPO):", mode)
    print(f"Inference Time: {end - start:.4f} seconds")
