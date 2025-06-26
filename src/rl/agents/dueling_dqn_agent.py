import torch.nn as nn
from .dqn_agent import DQNAgent

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(128, action_dim))
    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))

class DuelingDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        # Use DuelingQNetwork for both q_net and target_net
        self.q_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        super().__init__(state_dim, action_dim, **kwargs)
        self.q_net = self.q_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict()) 