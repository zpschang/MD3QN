import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

plt.rcParams.update({'font.size': 14})
plt.gcf().subplots_adjust(bottom=0.15)

data_dir = 'data-rebuttal'


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


envs = ['AirRaid', 'Asteroids', 'Gopher', 'MsPacman', 'Pong', 'UpNDown']
alg_names = ['HRA', 'MD3QN', 'MMDQN']
plot_alg_names = ['HRA', 'MD3QN', 'MMDQN']
plot_iterations = [99, 99, 99, 99, 99, 99]


def get_latest_path(env, alg_name, seed):
    if alg_name in ['HRA']:
        path = f'{data_dir}/{env}_{alg_name}_icml_seed-{seed}/logs'
    elif alg_name == 'MMDQN':
        path = f'data-mmdqn/dopamine_runs/{env}_{alg_name}_icml_seed-{seed}/logs'
    elif alg_name == 'MD3QN':
        path = f'{data_dir}/{env}_MMDQNND_icml_seed-{seed}_network-v21_bw-v3_kscale-v11/logs'
    elif alg_name == 'HADQN':
        path = f'{data_dir}/{env}_{alg_name}_icml_seed-0/logs'

    files = os.listdir(path)
    files_int = [int(s[4:]) for s in files]
    latest_file = sorted(files_int, reverse=True)[0]
    latest_path = f'{path}/log_{latest_file}'
    return latest_path


def read(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def get_eval_return(data):
    num_iterations = len(data.keys())
    eval_returns = []
    for iteration in range(num_iterations):
        data_iter = data[f'iteration_{iteration}']
        eval_returns.append(data_iter['eval_average_return'][0])
    return eval_returns


COLORS = ['slategray', 'tomato', 'dodgerblue', 'forestgreen', 'purple', 'orangered', 'black']


def plot_main():
    fig, axes = plt.subplots(2, 3, figsize=(45 / 2, 9 * 2))
    plot_num = 0
    lines = []
    for plot_iteration, env in zip(plot_iterations, envs):
        plot_x, plot_y = plot_num // 3, plot_num % 3
        all_eval_datas = []
        all_eval_datas_std = []
        all_eval_datas_min, all_eval_datas_max = [], []
        for alg_name in alg_names:
            eval_return_list = []
            min_len = 200
            for seed in range(3):
                path = get_latest_path(env, alg_name, seed)
                data = read(path)
                eval_returns = get_eval_return(data)
                eval_return_list.append(eval_returns)
                if len(eval_returns) < min_len:
                    min_len = len(eval_returns)

            for i in range(len(eval_return_list)):
                eval_return_list[i] = eval_return_list[i][:min_len]

            eval_return_list = np.stack(eval_return_list)
            all_eval_datas.append(np.mean(eval_return_list, axis=0))
            all_eval_datas_std.append(np.std(eval_return_list, axis=0))
            all_eval_datas_min.append(np.min(eval_return_list, axis=0))
            all_eval_datas_max.append(np.max(eval_return_list, axis=0))

        # plot_iteration = min(len(eval_data) for eval_data in all_eval_datas)
        # all_eval_datas = [eval_data[:plot_iteration] for eval_data in all_eval_datas]
        print(axes)
        print(plot_x, plot_y)
        axe = axes[plot_x, plot_y]
        axe.set_facecolor('whitesmoke')
        axe.grid(linestyle='--', color='darkgray', linewidth=1.5)
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        if plot_y == 0:
            axe.set_ylabel("Average Score", fontsize=38)
        axe.set_xlabel("Training Frames (millions)", fontsize=34)
        axe.set_title(env, fontsize=50)
        axe.tick_params(labelsize=16)
        for idx, eval_data in enumerate(all_eval_datas):
            eval_data = smooth(eval_data, 5)
            x_idx = []
            for i in range(len(eval_data)):
                x_idx.append(i)
            l = axe.plot(eval_data, color=COLORS[idx], label=plot_alg_names[idx], linewidth=4)
            # axes[plot_cnt].fill_between(x_idx, eval_data - all_eval_datas_std[idx], eval_data + all_eval_datas_std[idx], color=COLORS[idx], alpha=.3)
            # print(all_eval_datas_min[idx] - all_eval_datas_max[idx])
            axe.fill_between(x_idx,
                             all_eval_datas_min[idx] * 0.8 + all_eval_datas_max[idx] * 0.2,
                             all_eval_datas_min[idx] * 0.2 + all_eval_datas_max[idx] * 0.8,
                             color=COLORS[idx], alpha=.3)
            lines.append((l, plot_alg_names[idx]))
        plot_num += 1

    fig.subplots_adjust(top=0.95, left=0.08, right=0.98, bottom=0.15)

    leg = axes[1, 1].legend(loc='lower center', prop={'size': 38}, ncol=4,
                            bbox_to_anchor=(0.58, -0.4))
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    fig.align_labels()
    plt.savefig('Atari_HRA_MMDQN.pdf')


def test():
    path = get_latest_path('Gopher', 'HRA')
    data = read(path)
    eval_returns = get_eval_return(data)
    plt.plot(eval_returns)
    plt.show()


if __name__ == '__main__':
    plot_main()