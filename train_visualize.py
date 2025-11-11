import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from DQN_model import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PERDQNAgent, RainbowDQNAgent

def train_agent(agent_class, agent_name, env_name="MountainCar-v0", episodes=1000):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = agent_class(state_size, action_size)
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"{agent_name} - Episode {episode + 1}, Average Reward (last 100): {avg_reward:.2f}")
    
    env.close()
    return rewards_history

def smooth_data(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(all_results):
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, rewards) in enumerate(all_results.items()):
        sampled_rewards = rewards[::50]
        episodes = list(range(0, len(rewards), 50))
        
        if len(sampled_rewards) > 20:
            smoothed_rewards = smooth_data(sampled_rewards, window_size=5)
            smoothed_episodes = episodes[:len(smoothed_rewards)]
            plt.plot(smoothed_episodes, smoothed_rewards, label=name, color=colors[i], linewidth=2)
        else:
            plt.plot(episodes, sampled_rewards, label=name, color=colors[i], linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Algorithms Performance on MountainCar-v0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    agents = {
        'DQN': DQNAgent,
        'DoubleDQN': DoubleDQNAgent,
        'DuelingDQN': DuelingDQNAgent,
        'PERDQN': PERDQNAgent,
        'RainbowDQN': RainbowDQNAgent
    }
    
    all_results = {}
    
    for name, agent_class in agents.items():
        print(f"\n=== Training {name} ===")
        rewards = train_agent(agent_class, name, episodes=1000)
        all_results[name] = rewards
    
    print("\n=== Final Performance Comparison (last 100 episodes) ===")
    for name, rewards in all_results.items():
        avg_final = np.mean(rewards[-100:])
        print(f"{name}: {avg_final:.2f}")
    
    plot_results(all_results)

if __name__ == "__main__":
    main()
