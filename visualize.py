import json
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth_data(data, window_size=50):
    """使用移动平均平滑数据"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results():
    # 读取结果
    with open('results/all_results.json', 'r') as f:
        all_results = json.load(f)
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, rewards) in enumerate(all_results.items()):
        # 每50个episode取样一次
        sampled_rewards = rewards[::50]
        episodes = list(range(0, len(rewards), 50))
        
        # 如果数据点太多，进行平滑处理
        if len(sampled_rewards) > 20:
            smoothed_rewards = smooth_data(sampled_rewards, window_size=5)
            # 调整x轴以匹配平滑后的数据长度
            smoothed_episodes = episodes[:len(smoothed_rewards)]
            plt.plot(smoothed_episodes, smoothed_rewards, label=name, color=colors[i % len(colors)], linewidth=2)
        else:
            plt.plot(episodes, sampled_rewards, label=name, color=colors[i % len(colors)], linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Algorithms Performance on MountainCar-v0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印最终性能比较
    print("\n=== Final Performance Comparison (last 100 episodes) ===")
    for name, rewards in all_results.items():
        if len(rewards) >= 100:
            avg_final = np.mean(rewards[-100:])
            print(f"{name}: {avg_final:.2f}")

if __name__ == "__main__":
    if not os.path.exists('results/all_results.json'):
        print("Please run train.py first to generate training results!")
    else:
        plot_results()我