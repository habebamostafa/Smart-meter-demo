import numpy as np
import pickle

class QTableAgent:
    def __init__(self, q_table_path, bins_path):
        
        self.q_table = np.load(q_table_path)
        
        with open(bins_path, 'rb') as f:
            self.bins = pickle.load(f)

        self.keys = ["signal_strength_RF", "latency_RF", "signal_strength_plc", "latency_plc"]

    def _discretize_state(self, state):
        discrete_state = tuple(
            np.digitize(state[i], self.bins[key]) for i, key in enumerate(self.keys)
        )
        return discrete_state

    def act(self, state):
        state_discrete = self._discretize_state(state)
        return np.argmax(self.q_table[state_discrete])


q_agent = QTableAgent("trained_models/q_table.npy", "trained_models/q_bins.pkl")

def select_comm_mode_qtable(signal_rf, latency_rf, signal_plc, latency_plc):
    state = np.array([signal_rf, latency_rf, signal_plc, latency_plc], dtype=np.float32)
    action = q_agent.act(state)
    return "RF" if action == 0 else "PLC"

if __name__ == "__main__":
    # mode = select_comm_mode_qtable(-94.14, 46.6, -87.80, 15.9)
    mode = select_comm_mode_qtable(-87.143159, 46.598387, -87.806131, 115.937354)
    print("Recommended Communication Mode (Q-Table):", mode)
