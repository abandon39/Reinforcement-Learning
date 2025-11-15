import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from ExperienceReplayBuffer import ReplayBuffer
from neuralNetwork import DQN_Network

class DQNAgent:
    def __init__(self, state_dim, action_dim, epsilon_steps=20000, learning_rate=0.001, gamma=0.99):
        #初始化DQN参数
        self.state_dim = state_dim  #状态空间维数
        self.action_dim = action_dim    #动作空间维数
        self.main_buffer = ReplayBuffer(500000)
        self.success_buffer = ReplayBuffer(100000)
        self.batch_size = 64    #采样大小
        self.success_batch_percent = 0.8    #成功缓冲区采样比例
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_max = 1
        self.epsilon = self.epsilon_max
        self.epsilon_min = 0.005
        self.epsilon_decay = epsilon_steps    #ε线性衰减，掉到ε_min的episode
        self.steps = 0

        # 设置重复k次探索
        self.repeat_action = None
        self.repeat_action_number = 0

        #初始化两个网络
        self.main_network = DQN_Network(state_dim, action_dim)
        self.target_network = DQN_Network(state_dim, action_dim)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

        #同步目标网络参数
        self.update_target_network()    #初始化类就直接执行
    
    def update_epsilon(self):
        if self.epsilon >=  self.epsilon_min :
            self.epsilon -= ( self.epsilon_max - self.epsilon_min ) / ( self.epsilon_decay )

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    # 添加经验
    def add_experience(self, buffer, state, action, reward, next_state, done):
        # print(f"state={state},action={action},reward={reward},next_state={next_state},done={done}")
        if buffer == 'main_buffer':
            self.main_buffer.add_experience(state, action, reward, next_state, done)
        elif buffer == 'success_buffer':
            self.success_buffer.add_experience(state, action, reward, next_state, done)

    # #使用ε-greedy选择动作
    # def choose_action(self,state):
    #     # 选择探索，随便选动作空间里的一个动作
    #     if np.random.rand() <= self.epsilon_max:
    #         return random.randrange(self.action_dim)
    #     #选择利用，选q hat最大的
    #     #unsqueeze解读：unsqueeze就是将其升维变成二维矩阵，0表示在之前填充维度，1表示在之后填充维度
    #     #比如(3)👈3维行向量，unsqueeze(0)变成(1,3)，1行3列矩阵，在3之前的维度插入一个维度。
    #     state_tensor = torch.FloatTensor(state).unsqueeze(0)
    #     q_values = self.target_network(state_tensor)  #计算q value
    #     return torch.argmax(q_values).item()    #返回最大的q所对应的action
    
    #使用ε-greedy选择动作
    def choose_action(self,state):
        # 选择探索
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim),np.random.randint(5,13)    #随便选动作空间里的动作
        #选择利用，选q hat最大的
        #unsqueeze解读：unsqueeze就是将其升维变成二维矩阵，0表示在之前填充维度，1表示在之后填充维度
        #比如(3)👈3维行向量，unsqueeze(0)变成(1,3)，1行3列矩阵，在3之前的维度插入一个维度。
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # 传统DQN
        q_values = self.main_network(state_tensor)  #计算q value
        return torch.argmax(q_values).item(),1    #返回最大的q所对应的action和执行次数
        # Double DQN：用
        #
    
    #采样数据并训练函数
    def train(self):
        if self.main_buffer.size < self.batch_size * 10:
            return  #没有经验，没法训练
        #有经验，训练
        #采样，根据success_batch_persent选择
        if self.success_buffer.size < self.batch_size * self.success_batch_percent :
            state, actions, rewards, next_state, done = self.main_buffer.sample(self.batch_size)
        else :
            state, actions, rewards, next_state, done = self.main_buffer.sample(int(self.batch_size * (1 - self.success_batch_percent)))
            state_success, actions_success, rewards_success, next_state_success, done_success = self.success_buffer.sample(int(self.batch_size * self.success_batch_percent))
            state = np.concatenate([state, state_success])
            actions = np.concatenate([actions, actions_success])
            rewards = np.concatenate([rewards, rewards_success])
            next_state = np.concatenate([next_state, next_state_success])
            done = np.concatenate([done, done_success])
        
        state = torch.FloatTensor(state)
        actions = torch.IntTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(done)
        next_state = torch.FloatTensor(next_state)

        #计算当前q值
        # 这句代码的意思是，我们将state也就是positions,speed这个二维矩阵传入main_network进行前向传播
        # 比如我们的state每行表示position1,speed1,每次神经网络输入该行，然后输出一个行向量，得到q hat=(qhat_action1,qhat_action2,...)
        # 将结果qhat使用gather获取actions的对应q hat，并将这个1列n行的二维张量赋值给current_q_values
        # 其中gather(dim=1)表示在qhat的每行按列进行筛选。
        #（qhat是二维矩阵，每行是某个state的所有reward，用下标来表示这是第几个action对应的reward）
        #（actions是行向量使用unsqeeze(1)将其变为二维的n行1列矩阵）
        current_q_values = self.main_network(state).gather(1, actions.unsqueeze(1)).squeeze()

        #计算目标Q值
        with torch.no_grad():
            # max表示在1维度上取最大值，返回(value,index)，[0]表示只要最大值不要索引
            next_q_values = self.target_network(next_state).max(1)[0]
            # 使用贝尔曼最优公式计算TD target，即用下一时刻的q hat最大值来作为q的估计。
            # 由于使用了经验回放，这里的next_state都是已经知道的。
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            # print(f"next_q_values={next_q_values}")
        
        # 使用MSE计算损失，MSEloss只能接受向量，不能接受二维的矩阵
        loss = F.mse_loss(current_q_values, target_q_values)

        #反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=10)
        self.optimizer.step()

        #更新目标网络
        self.steps += 1
        if self.steps % 50 == 0:
            self.update_target_network()
        
        #更新epsilon
        self.update_epsilon()

        # 打印一些调试参数
        # if self.steps % 1500 == 0:
        #     print(f"Step {self.steps}:")
        #     print(f"  Loss: {loss.item():.4f}")
        #     print(f"  Epsilon: {self.epsilon:.3f}")
        #     print(f"  Avg Q: {current_q_values.mean().item():.3f}")
        #     print(f"  Buffer size: {self.main_buffer.size}")
        
    def save_model(self, file_path):
        # pass
        torch.save(self.main_network.state_dict(), file_path)
    
    def load_model(self, file_path):
        # pass
        self.main_network.load_state_dict(torch.load(file_path))
        self.update_target_network()  # 同步到目标网络
