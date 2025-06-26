from .prioritized_replay_dqn_agent import PrioritizedReplayDQNAgent
from .dueling_dqn_agent import DuelingQNetwork
import torch
import torch.nn as nn

class RainbowDQNAgent(PrioritizedReplayDQNAgent):
    def __init__(self, state_dim, action_dim, alpha=0.6, beta=0.4, **kwargs):
        # Use DuelingQNetwork for both q_net and target_net
        self.q_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        super().__init__(state_dim, action_dim, alpha=alpha, beta=beta, **kwargs)
        self.q_net = self.q_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done, indices, weights = self.buffer.sample(self.batch_size, beta=self.beta)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: action selection from q_net, value from target_net
            next_q_online = self.q_net(next_state)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_net(next_state).gather(1, next_actions).squeeze(1)
            target = reward + self.gamma * next_q_target * (1 - done)
        td_error = q_values - target
        loss = (weights * td_error ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        new_priorities = td_error.abs().detach().cpu().numpy() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 