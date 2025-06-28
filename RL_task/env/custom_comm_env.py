import gym
from gym import spaces
import numpy as np
import pandas as pd
import random  

class CustomCommEnv(gym.Env):
    """
    Custom Environment for Communication Mode Switching
    Actions:
        0 = Choose RF
        1 = Choose PLC
    Observations:
        [signal_strength_RF, latency_RF, signal_strength_plc, latency_plc]
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, debug=False):
        super(CustomCommEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.debug = debug

        low_obs = np.array([-120, 0, -120, 0], dtype=np.float32)
        high_obs = np.array([0, 1000, 0, 1000], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # (0 = RF, 1 = PLC)
        self.action_space = spaces.Discrete(2)

        self.last_action = None
        self.last_reward = None

        self.previous_action = None
        self.stay_count = 0

    def reset(self):
        self.current_step = 0
        self.last_action = None
        self.last_reward = None
        self.previous_action = None
        self.stay_count = 0
        return self._get_state()

    def step(self, action):
        row = self.data.iloc[self.current_step]

        if action == 0:  # RF
            signal = row['signal_strength_RF'] + random.uniform(-2.0, 2.0)
            latency = row['latency_RF'] + random.uniform(-2.0, 2.0)
            energy = row['Energy_kWh_rf']
        else:  # PLC
            signal = row['signal_strength_plc'] + random.uniform(-1.0, 1.0)
            latency = row['latency_plc'] + random.uniform(-1.0, 1.0)
            energy = row['Energy_kWh']

        # reward = 1.0 if signal >= -85 else -1.0
        # reward -= 0.0005 * latency
        # reward -= 1.3 * energy
        reward = 1.0 if signal >= -85 else -1.0
        reward -= (latency / 200)  
        reward -= (energy / 0.012)   


        if self.previous_action == action:
            self.stay_count += 1
        else:
            self.stay_count = 0  

        if self.stay_count > 5:
            stay_penalty = -0.1 * self.stay_count
            reward += stay_penalty
        else:
            stay_penalty = 0.0

        switch_penalty = 0.0
        if self.previous_action is not None and action != self.previous_action:
            switch_penalty = -1.0  
            reward += switch_penalty

        self.previous_action = action
        self.last_action = action
        self.last_reward = reward

        if self.debug:
            print(f"[Step {self.current_step}] Action: {'RF' if action==0 else 'PLC'} | "
                  f"Signal: {signal:.2f}, Latency: {latency:.2f}, "
                  f"Energy: {energy:.6f}, Reward: {reward:.3f}, "
                  f"Stay penalty: {stay_penalty:.2f}, Switch penalty: {switch_penalty:.2f}")

        self.current_step += 1
        done = self.current_step >= len(self.data)

        next_state = self._get_state() if not done else np.zeros(4, dtype=np.float32)

        return next_state, reward, done, {}

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        return np.array([
            row['signal_strength_RF'],
            row['latency_RF'],
            row['signal_strength_plc'],
            row['latency_plc']
        ], dtype=np.float32)

    def render(self, mode="human"):
        if self.last_action is not None:
            action_str = "RF" if self.last_action == 0 else "PLC"
            print(f"[Step {self.current_step}] Action: {action_str}, Reward: {self.last_reward:.4f}")
