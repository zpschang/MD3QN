import gym
import numpy as np
from dopamine.discrete_domains import atari_lib
import pickle

class RewardLogger(object):
    def __init__(self):
        self.count = {}
        self.reward_list = []

    def load(self, load_file):
        with open(load_file, 'rb') as f:
            obj = pickle.load(f)
        self.count, self.reward_list = obj['count'], obj['reward_list']

    def log(self, reward):
        if reward not in self.count:
            self.count[reward] = 1
        else:
            self.count[reward] += 1
        self.reward_list = sorted(set(self.count.keys()).difference({0.0}))

    def export(self, export_file):
        obj = {'count': self.count, 'reward_list': self.reward_list}
        with open(export_file, 'wb') as f:
            pickle.dump(obj, f)

    def sparsity(self):
        num_zeros = total = 0
        if 0.0 in self.count:
            num_zeros = self.count[0.0]
        for k, v in self.count.items():
            total += v
        return num_zeros / (total + 1e-5)

    def entropy(self):
        total_nonzero = 0
        for k, v in self.count.items():
            if k != 0.0:
                total_nonzero += v
        if total_nonzero == 0:
            return 0
        entropy = 0.0
        for k, v in self.count.items():
            if k != 0.0:
                entropy += -(v / total_nonzero) * np.log(v / total_nonzero)
        return entropy

    def __repr__(self):
        obj = {'entropy': self.entropy(),
               'sparsity': self.sparsity(),
               'count': self.count,
        }
        return obj.__repr__()

    @property
    def reward_dim(self):
        return len(self.reward_list) + 1

reward_logger = RewardLogger()

class MultiRewardEnv(object):
    def __init__(self, environment: atari_lib.AtariPreprocessing):
        self.environment = environment

    @property
    def observation_space(self):
        return self.environment.observation_space

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def reset(self):
        return self.environment.reset()

    def render(self, mode):
        return self.environment.render(mode)

    def step(self, action):
        observation, accumulated_reward, is_terminal, info = self.environment.step(action)
        try:
            index = reward_logger.reward_list.index(accumulated_reward)
        except ValueError:
            index = -1
        index += 1
        reward_list = [0.0 for _ in range(len(reward_logger.reward_list) + 1)]
        reward_list[index] = accumulated_reward
        return observation, reward_list, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        return self.environment._fetch_grayscale_observation(output)

    def _pool_and_resize(self):
        return self.environment._pool_and_resize()

    @property
    def game_over(self):
        return self.environment.game_over
    
if __name__ == '__main__':
    rewardLogger = RewardLogger()
    for _ in range(10):
        rewardLogger.log(0.0)
    for _ in range(2):
        rewardLogger.log(1.0)
    for _ in range(7):
        rewardLogger.log(-1.0)
    print(rewardLogger.__repr__())
    rewardLogger.export('test.pkl')