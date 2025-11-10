import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
from dqn_model import RepeatActionDQN
from stable_baselines3 import DQN

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

def create_gif_from_episode():
    """创建成功登顶的GIF动画 - 选择步数最少的一次"""
    print("正在生成GIF动画...")
    
    # 创建环境（使用rgb_array渲染模式）
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    model_mountaincar = DQN.load("mountaincar")
    
    best_frames = None
    best_steps = float('inf')
    max_attempts = 10  # 最多尝试10次
    
    for attempt in range(max_attempts):
        obs, _ = env.reset()
        episode_frames = []
        episode_steps = 0
        success = False
        
        for step in range(200):  # 最多200步
            # 渲染当前帧
            frame = env.render()
            # 转换为PIL Image
            img = Image.fromarray(frame)
            episode_frames.append(img)
            
            # 预测动作
            action, _ = model_mountaincar.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_steps += 1
            
            if done:
                # 检查是否成功登顶
                if obs[0] >= 0.5:
                    success = True
                break
        
        # 如果成功登顶，检查是否是最佳表现
        if success:
            print(f"第{attempt+1}次尝试成功登顶！步数：{episode_steps}")
            
            # 如果这次步数更少，更新最佳记录
            if episode_steps < best_steps:
                best_steps = episode_steps
                best_frames = episode_frames
                print(f"更新最佳记录：{best_steps} 步")
        else:
            print(f"第{attempt+1}次尝试未成功登顶")
    
    env.close()
    
    if best_frames:
        # 保存为GIF
        gif_path = "mountaincar_success.gif"
        
        # 使用Pillow保存GIF
        best_frames[0].save(
            gif_path,
            save_all=True,
            append_images=best_frames[1:],
            duration=66,  # 约15fps (1000/15 ≈ 66ms)
            loop=0,       # 无限循环
            optimize=True  # 优化文件大小
        )
        
        print(f"GIF动画已保存至: {gif_path}")
        print(f"最佳表现：{best_steps} 步，共 {len(best_frames)} 帧")
        print(f"GIF尺寸: {best_frames[0].size}")
        
        return gif_path, best_steps
    else:
        print("在多次尝试后仍未成功登顶，无法生成GIF")
        return None, None

def test_model_with_gif():
    """测试模型并生成GIF"""
    print("正在测试模型并生成GIF动画...")
    
    # 生成GIF
    gif_path, best_steps = create_gif_from_episode()
    
    if gif_path and os.path.exists(gif_path):
        print(f"成功生成GIF文件: {gif_path}")
        print(f"最佳登顶步数: {best_steps} 步")
        file_size = os.path.getsize(gif_path) / 1024
        print(f"文件大小: {file_size:.2f} KB")
    else:
        print("GIF生成失败")

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
    
    # 4. 测试模型并生成GIF
    env.close()
    test_model_with_gif()
    
    print("任务完成！已生成价值函数可视化图表和GIF动画")

if __name__ == "__main__":
    main()