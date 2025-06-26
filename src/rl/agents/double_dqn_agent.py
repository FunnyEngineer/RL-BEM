from .dqn_agent import DQNAgent
import torch
import torch.nn as nn
import numpy as np

class DoubleDQNAgent(DQNAgent):
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: action selection from q_net, value from target_net
            next_q_online = self.q_net(next_state)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_net(next_state).gather(1, next_actions).squeeze(1)
            target = reward + self.gamma * next_q_target * (1 - done)
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 