import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import torch

class DualReplayBuffer:
    def __init__(self, main_buffer_size=50000, success_buffer_size=5000, success_ratio=0.3):
        self.main_buffer = []
        self.success_buffer = []

        self.main_buffer_size = main_buffer_size
        self.success_buffer_size = success_buffer_size
        self.success_sampling_ratio = success_ratio

        self.current_episode = []

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.current_episode.append(transition)

        if done:
            # 判断是否为成功 episode（done而且位置>=0.5）
            last_state = self.current_episode[-1][3]
            is_success = done and (last_state[0] >= 0.5)

            # 把整个 episode 加入主缓冲池
            for trans in self.current_episode:
                self.main_buffer.append(trans)
                if len(self.main_buffer) > self.main_buffer_size:
                    self.main_buffer.pop(0)

            # 如果是成功 episode，把整个 episode 也加入成功缓冲池
            if is_success:
                for trans in self.current_episode:
                    self.success_buffer.append(trans)
                    if len(self.success_buffer) > self.success_buffer_size:
                        self.success_buffer.pop(0)

            # 清空当前 episode 缓存
            self.current_episode = []

    def sample(self, batch_size):
        success_batch_size = int(batch_size * self.success_sampling_ratio)
        indices = []

        # 从成功缓冲池采样
        if len(self.success_buffer) > 0:
            success_indices = np.random.choice(
                len(self.success_buffer),
                size=min(success_batch_size, len(self.success_buffer)),
                replace=False
            )
            indices.extend(('success', idx) for idx in success_indices)

        # 从主缓冲池补全
        remaining_size = batch_size - len(indices)
        if remaining_size > 0 and len(self.main_buffer) > 0:
            main_indices = np.random.choice(
                len(self.main_buffer),
                size=min(remaining_size, len(self.main_buffer)),
                replace=False
            )
            indices.extend(('main', idx) for idx in main_indices)

        # 构建 batch
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for buffer_type, idx in indices:
            if buffer_type == 'success':
                trans = self.success_buffer[idx]
            else:
                trans = self.main_buffer[idx]
            states.append(trans[0])
            actions.append(trans[1])
            rewards.append(trans[2])
            next_states.append(trans[3])
            dones.append(trans[4])

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
class RepeatActionDQN(DQN):
    """
    改进探索策略：连续动作探索（降低随机性，提升动作连贯性）
    逻辑：随机选方向（左/右）→ 连续执行N步 → 再随机换方向
    """
    def __init__(self, *args, **kwargs):
        # 提取新增的探索策略参数（默认值适配MountainCar）
        self.min_continuous_steps = kwargs.pop('min_continuous_steps', 5)         # 最小连续步数
        self.max_continuous_steps = kwargs.pop('max_continuous_steps', 12)        # 最大连续步数
        
        # 原有参数提取
        self.success_ratio = kwargs.pop('success_ratio', 0.3)
        self.success_buffer_size = kwargs.pop('success_buffer_size', 5000)
        
        super().__init__(*args, **kwargs)
        self.useful_actions = [0, 2]  # 只探索左/右（排除不动）
        
        # 新增探索状态变量
        self.current_explore_action = None  # 当前探索的动作
        self.remaining_continuous_steps = 0  # 该动作还需连续执行的步数

    def _create_buffer(self, buffer_size: int, observation_space, action_space):
        return DualReplayBuffer(
            main_buffer_size=buffer_size,
            success_buffer_size=self.success_buffer_size,
            success_ratio=self.success_ratio
        )

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # 贪心分支：清空探索状态，使用最优动作
        if deterministic or np.random.rand() > self.exploration_rate:
            self.current_explore_action = None
            self.remaining_continuous_steps = 0
            return super().predict(observation, state=state,
                                episode_start=episode_start,
                                deterministic=True)

        # 探索分支：连续执行同一动作N步，再随机换方向
        if self.remaining_continuous_steps > 0:
            # 还有剩余步数：继续执行当前探索动作
            action_to_take = self.current_explore_action
            self.remaining_continuous_steps -= 1
        else:
            # 无剩余步数：随机选新动作+随机确定连续步数（动态范围）
            self.current_explore_action = np.random.choice(self.useful_actions)  # 只选左/右
            self.remaining_continuous_steps = np.random.randint(self.min_continuous_steps, self.max_continuous_steps + 1)
            action_to_take = self.current_explore_action

        return np.array([action_to_take]), state

# =================== 训练函数 ===================
def train_dqn_agent():
    print("正在训练 DQN（ε-greedy + 重复上次动作）…")
    env = gym.make('MountainCar-v0')

    model = RepeatActionDQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        success_buffer_size=5000,    # 成功缓冲池大小
        success_ratio=0.3,           # 成功经验采样比例
        batch_size=32,
        gamma=0.99,
        target_update_interval=100,
        exploration_fraction=0.5,   # ε 从 1→0.05 的步数占比
        exploration_initial_eps=1.0,
        exploration_final_eps=0.005,
        verbose=1,
        device="cpu"
    )
    model.learn(total_timesteps=200000)
    print("训练完成！")
    return model, env

def extract_value_function(model, grid_size=50):
    """提取价值函数 V(s) = max Q(s,a)"""
    print("正在提取价值函数...")
    
    positions = np.linspace(-1.2, 0.6, grid_size)
    velocities = np.linspace(-0.07, 0.07, grid_size)
    value_grid = np.zeros((grid_size, grid_size))
    
    # 确保模型在评估模式
    model.policy.set_training_mode(False)
    
    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            state = np.array([pos, vel])
            
            # 使用Q网络计算V(s) = max Q(s,a)
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                # 使用policy.q_net而不是model.q_net
                q_values = model.policy.q_net(state_tensor).numpy().flatten()
                value_grid[i, j] = np.max(q_values)
    
    return positions, velocities, value_grid

def plot_heatmap(positions, velocities, value_grid):
    """绘制二维热力图"""
    plt.figure(figsize=(10, 6))
    plt.imshow(value_grid.T, 
               extent=[-1.2, 0.6, -0.07, 0.07], 
               origin='lower', 
               aspect='auto', 
               cmap='viridis')
    plt.colorbar(label='Value V(s)')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Value Function V(s) = max Q(s,a)')
    plt.savefig('value_function.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_surface(positions, velocities, value_grid):
    """绘制三维曲面图"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(positions, velocities)
    surf = ax.plot_surface(X, Y, value_grid.T, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value V(s)')
    ax.set_title('Value Function - 3D Surface')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('value_function_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("开始执行强化学习价值函数可视化任务")
    print("=" * 50)
    
    # 1. 训练智能体
    model, env = train_dqn_agent()
    model.save("mountaincar")
    
    # 2. 提取价值函数
    positions, velocities, value_grid = extract_value_function(model)
    
    # 3. 可视化
    plot_heatmap(positions, velocities, value_grid)
    plot_3d_surface(positions, velocities, value_grid)
    env.close()
    model_test=gym.make("MountainCar-v0",render_mode="human")
    model_mountaincar=DQN.load("mountaincar")
    while True:
        obs, _ = model_test.reset()
        episode_steps = 0
        success = False
        for step in range(200):  # 最多200步（环境默认超时阈值）
            action, _ = model_mountaincar.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = model_test.step(action)
            episode_steps += 1
            if done:
                success = True
                break
        if success:
            print(f"成功渲染！本次episode步数：{episode_steps}")
            break  # 成功后停止循环
    model_test.close()
    print("任务完成！已生成价值函数可视化图表")

if __name__ == "__main__":
    main()