import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

class PPOAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, eps_clip=0.1, lr=1e-4):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs, _ = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, states, actions):
        probs, values = self.model(states)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_logprobs, torch.squeeze(values), entropy

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, memory):
        states = torch.tensor(np.array(memory['states']), dtype=torch.float32)
        actions = torch.tensor(memory['actions'], dtype=torch.int64)
        old_logprobs = torch.tensor(memory['logprobs'], dtype=torch.float32)
        rewards = torch.tensor(memory['rewards'], dtype=torch.float32)
        dones = torch.tensor(memory['dones'], dtype=torch.float32)

        with torch.no_grad():
            _, next_value = self.model(states[-1].unsqueeze(0))
            returns = self.compute_returns(rewards, dones, next_value)

        for _ in range(10):  # More training per episode
            logprobs, values, entropy = self.evaluate(states, actions)
            advantages = returns - values.detach()

            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * (returns - values).pow(2).mean() - \
                   0.001 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, path="trained_models/customed_ppo_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="trained_models/customed_ppo_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.eval()
