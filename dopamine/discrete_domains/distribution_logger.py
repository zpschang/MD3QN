import gin.tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import numpy as np

bws = [2 ** i for i in range(-8, 9)]
bws_large = [4 ** i for i in range(-8, 9)]

def compute_mmd(x, y, bws, use_scale=False):
    loss = 0.0
    l2_x_x = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
    l2_x_y = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)
    l2_y_y = ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)
    for bw in bws:
        l = 0.0
        l += np.exp(-l2_x_x / bw).mean()
        l += np.exp(-l2_y_y / bw).mean()
        l += -2 * np.exp(-l2_x_y / bw).mean()
        if use_scale:
            l *= np.sqrt(bw)
        loss += l
    return loss

class DistributionLogger(object):
    def __init__(self):
        self.logdir = None
        self.iteration = None

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_iteration(self, iteration):
        self.iteration = iteration

    def log_trajectory(self, time, state, action, reward, samples, idx):
        state = state[0].astype(np.uint8)
        path = os.path.join(self.logdir, 'evaluation_plots', f'iter-{self.iteration}', f'traj-{idx}')
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, 'primitive'), exist_ok=True)

        data = {
            'time': time,
            'state': state,
            'action': action,
            'reward': reward,
            'samples': samples
        }
        pickle_filename = os.path.join(path, f'primitive/{time:05d}.pickle')
        # with open(pickle_filename, 'wb') as file:
        #     pickle.dump(data, file)
        os.makedirs(os.path.join(path, 'ar'), exist_ok=True)
        with open(os.path.join(path, 'ar', f'{time:05d}-action-reward.txt'), 'w') as file:
            file.write(f'action: {action}\n')
            file.write(f'reward: {reward}\n')

        os.makedirs(os.path.join(path, 'state'), exist_ok=True)
        Image.fromarray(state[..., 1:]).save(os.path.join(path, 'state', f'{time:05d}-state.png'))

        reward_dim = samples.shape[1]
        # plot 2d joint distribution
        for d1 in range(reward_dim):
            for d2 in range(d1 + 1, reward_dim):
                sample_d1, sample_d2 = samples[:, d1], samples[:, d2]
                plt.scatter(sample_d1, sample_d2, alpha=0.5)

                os.makedirs(os.path.join(path, f'ret-d{d1}-d{d2}'), exist_ok=True)
                plt.savefig(os.path.join(path, f'ret-d{d1}-d{d2}', f'{time:05d}-return-dim{d1}-dim{d2}.png'), dpi=70)
                plt.clf()


    def log_evaluation(self, prediction_samples, mc_samples, episode_idx, num_steps=None):
        path = os.path.join(self.logdir, 'evaluation_plots', f'iter-{self.iteration}')
        os.makedirs(path, exist_ok=True)

        mmd = compute_mmd(prediction_samples, mc_samples, bws)
        mmd_scaled = compute_mmd(prediction_samples, mc_samples, bws, use_scale=True)
        mme_large = compute_mmd(prediction_samples, mc_samples, bws_large)

        print(f'logging into evaluation_plots, num_steps is {num_steps}')
        print(f'mmd is {mmd}, mmd_scaled is {mmd_scaled}, mmd_large is {mme_large}')
        data = {
            'prediction_samples': prediction_samples,
            'mc_samples': mc_samples,
            'num_steps': num_steps,
            'mmd': mmd,
            'mmd_scaled': mmd_scaled,
            'mme_large': mme_large
        }
        pickle_filename = os.path.join(path, f'episode-{episode_idx}.pickle')
        with open(pickle_filename, 'wb') as file:
            pickle.dump(data, file)

        reward_dim = prediction_samples.shape[1]
        # plot 2d joint distribution
        for d1 in range(reward_dim):
            for d2 in range(d1 + 1, reward_dim):
                prediction_d1, prediction_d2 = prediction_samples[:, d1], prediction_samples[:, d2]
                plt.scatter(prediction_d1, prediction_d2, alpha=0.5)

                mc_d1, mc_d2 = mc_samples[:, d1], mc_samples[:, d2]
                plt.scatter(mc_d1, mc_d2, alpha=0.5)

                plt.legend(['prediction', 'monte-carlo'])

                plt.savefig(os.path.join(path, f'scatter-dim{d1}-dim{d2}-episode-{episode_idx}.png'), dpi=70)
                plt.clf()


    def log(self, samples, next_samples, states, actions, rewards, next_states, losses, gamma):
        batch_size = samples.shape[0]
        # print('batch_size:', batch_size)
        for idx in range(batch_size):
            state, action, reward, next_state, loss = states[idx], actions[idx], rewards[idx], next_states[idx], losses[idx]
            path = os.path.join(self.logdir, 'plots', f'iter-{self.iteration}-idx-{idx}')
            os.makedirs(path, exist_ok=True)

            # write action and reward
            with open(os.path.join(path, 'action-reward.txt'), 'w') as file:
                file.write(f'action: {action}\n')
                file.write(f'reward: {reward}\n')
                file.write(f'loss: {loss}\n')

            # write state image
            Image.fromarray(state[..., 1:]).save(os.path.join(path, 'state.png'))
            Image.fromarray(next_state[..., 1:]).save(os.path.join(path, 'next-state.png'))

            # plot reward samples
            sample, next_sample = samples[idx], next_samples[idx]  # [sample_num, reward_dim]
            reward_dim = sample.shape[1]

            # plot 1d marginal distribution
            for d in range(reward_dim):
                sample_d = sample[:, d]
                plt.hist(sample_d, bins=20)
                plt.savefig(os.path.join(path, f'return-dim{d}.png'), dpi=70)
                plt.clf()

                next_sample_d = next_sample[:, d]
                plt.hist(next_sample_d, bins=20)
                plt.savefig(os.path.join(path, f'next-return-dim{d}.png'), dpi=70)
                plt.clf()

            # plot 2d joint distribution
            for d1 in range(reward_dim):
                for d2 in range(d1+1, reward_dim):
                    sample_d1, sample_d2 = sample[:, d1], sample[:, d2]
                    plt.scatter(sample_d1, sample_d2, alpha=0.5)

                    next_sample_d1, next_sample_d2 = next_sample[:, d1], next_sample[:, d2]
                    plt.scatter(next_sample_d1, next_sample_d2, alpha=0.5)

                    target_sample_d1 = gamma * next_sample_d1 + reward[d1]
                    target_sample_d2 = gamma * next_sample_d2 + reward[d2]

                    plt.scatter(target_sample_d1, target_sample_d2, alpha=0.5)

                    plt.legend(['z', 'zp', 'gamma * zp + r'])

                    plt.savefig(os.path.join(path, f'return-dim{d1}-dim{d2}.png'), dpi=70)
                    plt.clf()

            data = {
                'samples': samples,
                'next_samples': next_samples,
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'losses': losses,
                'gamma': gamma
            }

            pickle_filename = os.path.join(path, 'data.pkl')
            with open(pickle_filename, 'wb') as file:
                pickle.dump(data, file)


distribution_logger = DistributionLogger()