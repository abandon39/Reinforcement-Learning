import numpy as np
from stable_baselines3 import DQN
from dual_replay_buffer import DualReplayBuffer

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