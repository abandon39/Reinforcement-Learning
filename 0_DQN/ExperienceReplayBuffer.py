import numpy as np
import random

class ReplayBuffer:
    def __init__(self,capacity,observation_space_length=2):
        self.capacity = capacity
        # 存SARSA
        self.state = np.zeros((capacity,observation_space_length), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.long)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_state = np.zeros((capacity,observation_space_length), dtype=np.float32)
        self.done = np.zeros(capacity, np.bool_)
        self.index = 0 #当前插入/删除时的索引
        self.size = 0 #当前回放区的大小

    #添加一条经验
    def add_experience(self,state,action,reward,next_state,done):
        if self.index >= self.capacity:
            self.index = 0  #循环覆盖
        
        #添加数据，将矩阵的index位置改为提供的数据
        self.state[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_state[self.index] = next_state
        self.done[self.index] = done

        self.index += 1
        if self.size < self.capacity:
            self.size += 1

    #随机采样batch_size个单位，返回矩阵
    def sample(self,batch_size):
        samples = random.sample(range(self.size),batch_size)

        return (
            self.state[samples],
            self.actions[samples],
            self.rewards[samples],
            self.next_state[samples],
            self.done[samples]
        )
