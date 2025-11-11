import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Network.dqn_network import DuelingDQN
from Replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer

class RainbowDQNAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cpu")
        self.action_size = action_size
        
        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.memory = PrioritizedReplayBuffer(20000, alpha=0.5)
        self.batch_size = 64
        
        # 更稳定的探索
        self.epsilon_start, self.epsilon_end = 0.1, 0.005
        self.steps_done = 0
        self.total_steps = 50000
    
    def act(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  max(0, 1 - self.steps_done / (self.total_steps * 0.5))
        
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.policy_net(state).argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Rainbow核心：Dueling + Double + PER
        next_actions = self.policy_net(next_states).max(1)[1]
        next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q = rewards + (0.99 * next_q * ~dones)
        
        td_errors = (current_q - target_q).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        loss = (weights * (current_q - target_q.detach()).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())