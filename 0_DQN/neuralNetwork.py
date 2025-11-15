import torch.nn as nn

class DQN_Network(nn.Module):
    #定义网络结构，2个全连接隐藏层
    def __init__(self, state_dim, action_dim, hidden_size = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),  #接受输入层的2个输入
            nn.Tanh(),
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,action_dim)   #产生3个输出
        )

    #前向传播过程，输入的状态是x
    def forward(self,x):
        return self.network(x)