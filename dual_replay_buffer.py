import numpy as np

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