from .dqn_agent import DQNAgent
import numpy as np
import torch
import torch.nn as nn

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, indices, weights
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, alpha=0.6, beta=0.4, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.buffer = PrioritizedReplayBuffer(kwargs.get('buffer_size', 10000), alpha=alpha)
        self.beta = beta
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
            next_q = self.target_net(next_state).max(1)[0]
            target = reward + self.gamma * next_q * (1 - done)
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