import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqnMadebyMyself import DQNAgent

env_name = "MountainCar-v0"
# agent_class用于传入DQNAgent，env_name方便后续修改环境，epsidoes是训练轮次
def train_agent(agent_class, episodes=5000):
    env = gym.make(env_name)

    # 状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建agent
    agent = agent_class(state_dim, action_dim)

    # 设置数组存储reward
    rewards_history = []

    # 开始训练，
    for episode in range(episodes):
        # reset返回包含观察空间的元组（用state承接）和包含一些乱七八糟补充信息的dict，不要了
        state, _ = env.reset()
        total_reward = 0
        done = False

        whole_episode = []
        # 一直探索直到结束   
        while not done:
            # 执行一步
            action, frequency = agent.choose_action(state)
            # 设置数组保存整条episode
            for _ in range(frequency):
                #如果探索已经完成（195次以后选到了探索可能出现），则不再执行直接break
                if done:
                    break
                next_state, reward, terminated, truncated, _ = env.step(action)
                whole_episode.append([state, action, reward, next_state, done])
                # if terminated :
                #     reward += 100
                done = terminated or truncated

                # 保存奖励
                agent.add_experience('main_buffer', state, action, reward, next_state, done)

                #状态转移
                state = next_state
                total_reward += reward
        # 如果是正常完成的，则回溯整条路径并加入成功缓冲区
        if total_reward > -200:
            for items in whole_episode:
                agent.add_experience('success_buffer', items[0], items[1], items[2], items[3], items[4])
        # 每1步训练10次
        for i in range(10):
            agent.train()
        # 把奖励存起来用于打印
        rewards_history.append(total_reward)
        # 每10episode打印一次结果并学习一次
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"success_buffer_size = {agent.success_buffer.size}")
            print(f"DQN gets average reward(last 100 episodes):{avg_reward:.2f};epsilon={agent.epsilon};episode={episode}")

    env.close()
    return rewards_history, agent

# 画折线图，每50步采样一次
def plot_results(results:list):
    plt.figure(figsize=(12,8))

    # 遍历整个图，每50步采样一次，获取数据
    for i, rewards in enumerate(results):
        # 每50步采样
        episodes = list(range(0, len(results), 50))
        sample_rewards = results[::50]

    # 画图
    plt.plot(episodes, sample_rewards, linewidth=2)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.title("DQN performance on MountainCar-v0")
    # plt.legend()
    plt.show()

# 测试
def test(model_path = ""):
    env = gym.make(env_name,render_mode="human")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load_model(model_path)
    agent.epsilon = 0   #不探索

    for i in range(3):
        state,_ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action,_ = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
    env.close()


#设置主函数开始训练
def main():
    rewards,agent = train_agent(DQNAgent)
    agent.save_model("dqn_model")
    test("dqn_model")
    plot_results(rewards)

if __name__ == "__main__":
    main()