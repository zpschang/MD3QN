import numpy as np
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains.distribution_logger import distribution_logger
import random
import copy

class RewardLogger(object):
    def __init__(self):
        self.rewards = {}

    def log(self, reward):
        raise NotImplementedError

    def load(self, load_file):
        with open(load_file, 'r') as file:
            text = file.read()
            self.rewards = eval(text)

    def __repr__(self):
        return '...'

    @property
    def reward_dim(self):
        return len(self.rewards)

reward_logger = RewardLogger()

class MultiRewardEnv(object):
    def __init__(self, environment):
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

    def decompose(self, primitive_reward):
        remain_reward = primitive_reward
        reward_list = [0.0 for _ in range(reward_logger.reward_dim)]
        for index, (reward, cond) in enumerate(reward_logger.rewards.items()):
            is_valid = remain_reward >= reward and eval(cond.format(remain_reward - reward))
            if is_valid:
                remain_reward -= reward
                reward_list[index] += reward
        debug = False
        if primitive_reward != 0.0 and debug:
            print(f'primitive: {primitive_reward}')
            print(f'decompose: {reward_list}')
        return reward_list

    def step(self, action):
        observation, primitive_reward, is_terminal, info = self.environment.step(action)
        reward_list = self.decompose(primitive_reward)
        return observation, (reward_list, primitive_reward), is_terminal, info

    def _fetch_grayscale_observation(self, output):
        return self.environment._fetch_grayscale_observation(output)

    def _pool_and_resize(self):
        return self.environment._pool_and_resize()

    @property
    def game_over(self):
        return self.environment.game_over

    def monte_carlo_joint_return(self, obs, num_samples, initial_action, gamma=0.99,
                                 eval_policy=None, trained_agent=None):
        print('in atari monte carlo return')
        ale_state = self.environment.environment.clone_state()
        samples = []
        visualize_trajectory = False  # modified: 8-1
        for idx in range(num_samples):
            if idx == 0:
                print('start visualizing trajectory')
            self.environment.environment.restore_state(ale_state)
            time = 0
            episode_reward = np.zeros(reward_logger.reward_dim)
            while True:
                if time == 0:
                    action = initial_action
                    eval_policy.begin_episode(obs)
                else:
                    if eval_policy is None:
                        action = random.choice(range(reward_logger.reward_dim))
                    else:
                        action = eval_policy.step(reward, obs)

                state = eval_policy.state
                obs, reward, terminate, _ = self.step(action)

                if type(reward) is np.ndarray:
                    pass
                else:
                    reward = np.array(reward[0])

                if idx == 0 and visualize_trajectory:
                    # visualize the distribution prediction of "trained_agent" under the monte-carlo trajectory
                    particles = trained_agent.joint_value_distribution_stacked(state, action)
                    distribution_logger.log_trajectory(time, state, action, reward, particles, idx)

                # TODO: add configs for reward clipping
                reward = np.clip(reward, -1, 1)
                episode_reward += reward * (gamma ** time)
                time += 1
                if terminate:
                    break
            samples.append(episode_reward)
            if idx == 0:
                print('finished visualizing trajectory')
            print(f'{idx}/{num_samples}: total reward = {episode_reward}, length={time}')
        samples = np.stack(samples)
        print('mc samples:', samples.shape)
        print('mean:', samples.mean(axis=0))
        print('std:', samples.std(axis=0))
        print('min:', samples.min(axis=0))
        print('max:', samples.max(axis=0))
        return samples

    def monte_carlo_joint_return_middle(self, num_samples, initial_action, gamma=0.99,
                                 eval_policy=None, trained_agent=None):
        print('in atari monte carlo return')
        ale_state = self.environment.environment.clone_state()
        policy_state = copy.deepcopy(eval_policy.state)
        samples = []
        visualize_trajectory = True  # modified: 8-1
        for idx in range(num_samples):
            if idx == 0:
                print(f'Start visualizing {idx}-th Monte-Carlo trajectory')
                visualize_trajectory_this_iteration = True
            else:
                visualize_trajectory_this_iteration = False
            self.environment.environment.restore_state(ale_state)
            eval_policy.state = policy_state
            time = 0
            episode_reward = np.zeros(reward_logger.reward_dim)
            while True:
                if time == 0:
                    action = initial_action
                else:
                    action = eval_policy.step(reward, obs)

                state = eval_policy.state
                obs, reward, terminate, _ = self.step(action)

                if type(reward) is np.ndarray:
                    pass
                else:
                    reward = np.array(reward[0])

                if visualize_trajectory and visualize_trajectory_this_iteration:
                    # visualize the distribution prediction of "trained_agent" under the monte-carlo trajectory
                    particles = trained_agent.joint_value_distribution_stacked(state, action)
                    distribution_logger.log_trajectory(time, state, action, reward, particles, idx)

                reward = np.clip(reward, -1, 1)
                episode_reward += reward * (gamma ** time)
                time += 1
                if terminate:
                    break
            samples.append(episode_reward)
            if visualize_trajectory_this_iteration:
                print(f'Finished visualizing {idx}-th Monte-Carlo trajectory')
            print(f'{idx}/{num_samples}: total reward = {episode_reward}, length={time}')
        samples = np.stack(samples)
        print('mc samples:', samples.shape)
        print('mean:', samples.mean(axis=0))
        print('std:', samples.std(axis=0))
        print('min:', samples.min(axis=0))
        print('max:', samples.max(axis=0))
        return samples

if __name__ == '__main__':
    reward_logger.load('reward-compose/Gopher-reward.txt')
    env = MultiRewardEnv(None)

    def test(tests):
        for t in tests:
            print(t, env.decompose(t))

    print('=== Gopher ===')
    tests = [0.0, 20.0, 100.0, 120.0]
    test(tests)

    print('=== MsPacman ===')
    reward_logger.load('reward-compose/MsPacman-reward.txt')
    tests = [0.0, 1610.0, 250.0, 10.0, 50.0, 110.0, 310.0, 500.0]
    test(tests)

    print('=== UpNDown ===')
    reward_logger.load('reward-compose/UpNDown-reward.txt')
    tests = [0.0, 10.0, 100.0, 600.0, 410.0, 110.0, 510.0, 810.0]
    test(tests)

    print('=== Pong ===')
    reward_logger.load('reward-compose/Pong-reward.txt')
    tests = [-1.0, 0.0, 1.0]
    test(tests)

