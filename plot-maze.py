import matplotlib.pyplot as plt
import pickle

from matplotlib.pyplot import figure

kp = 'prediction_samples'
km = 'mc_samples'

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.figsize': [6.4, 4.8]})

plt.gcf().subplots_adjust(bottom=0.18)
# plt.gcf().subplots_adjust(top=0.95)
plt.gcf().subplots_adjust(left=0.18)


def read(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj


v0 = read('v0.pickle')   # pickle file in evaluation_plots for MultiRewardMaze-v0
v1 = read('v1.pickle')   # pickle file in evaluation_plots for MultiRewardMaze-v1
v2 = read('v2.pickle')   # pickle file in evaluation_plots for MultiRewardMaze-v2

COLORS = ['dodgerblue', 'orangered', 'dodgerblue', 'slategray', 'purple', 'orangered', 'black']
legend_list = ['MD3QN', 'ground truth']

fig, axes = plt.subplots(1, 4, figsize=(30, 9))

# v0-d0d1
axes[2].set_title('(c)', fontsize=50)
axes[2].set_facecolor('whitesmoke')
axes[2].grid(linestyle='--', color='darkgray', linewidth=1.5, zorder=0)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].scatter(v0[kp][:, 0], v0[kp][:, 1], alpha=0.7, label=legend_list[0], s=160, c=COLORS[0], zorder=10)
axes[2].scatter(v0[km][:, 0], v0[km][:, 1], alpha=0.7, label=legend_list[1], s=160, c=COLORS[1], zorder=10)
axes[2].set_xlabel('Return in orange', fontsize=38)
axes[2].set_ylabel('Return in blue', fontsize=38)

# plt.savefig('v0-d0d1.png', dpi=200)
# plt.clf()

# v0-d2d3
axes[3].set_facecolor('whitesmoke')
axes[3].grid(linestyle='--', color='darkgray', linewidth=1.5, zorder=0)
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)
axes[3].set_title('(d)', fontsize=50)
axes[3].scatter(v0[kp][:, 2], v0[kp][:, 3], alpha=0.7, label=legend_list[0], s=160, c=COLORS[0], zorder=10)
axes[3].scatter(v0[km][:, 2], v0[km][:, 3], alpha=0.7, label=legend_list[1], s=160, c=COLORS[1], zorder=10)
# axes[1].legend(['MD3QN', 'ground truth'], fontsize=16)
axes[3].set_xlabel('Return in green', fontsize=38)
axes[3].set_ylabel('Return in red', fontsize=38)

# plt.savefig('v0-d2d3.png', dpi=200)
# plt.clf()

# v1
axes[0].set_facecolor('whitesmoke')
axes[0].grid(linestyle='--', color='darkgray', linewidth=1.5, zorder=0)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].set_title('(a)', fontsize=50)
axes[0].scatter(v1[kp][:, 0], v1[kp][:, 1], alpha=0.7, label=legend_list[0], s=160, c=COLORS[0], zorder=10)
axes[0].scatter(v1[km][:, 0], v1[km][:, 1], alpha=0.7, label=legend_list[1], s=160, c=COLORS[1], zorder=10)
# axes[2].legend(['MD3QN', 'ground truth'], fontsize=16)
axes[0].set_xlabel('Return in green', fontsize=38)
axes[0].set_ylabel('Return in red', fontsize=38)

# plt.savefig('v1.png', dpi=200)
# plt.clf()

# v2
axes[1].set_facecolor('whitesmoke')
axes[1].grid(linestyle='--', color='darkgray', linewidth=1.5, zorder=0)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_title('(b)', fontsize=50)
# axes[3].legend(['MD3QN', 'ground truth'], fontsize=16)
axes[1].set_xlabel('Return in green', fontsize=38)
axes[1].set_ylabel('Return in red', fontsize=38)
axes[1].scatter(v2[kp][:, 0], v2[kp][:, 1], alpha=0.7, label=legend_list[0], s=160, c=COLORS[0], zorder=10)
axes[1].scatter(v2[km][:, 0], v2[km][:, 1], alpha=0.7, label=legend_list[1], s=160, c=COLORS[1], zorder=10)

for i in range(4):
    axes[i].tick_params(labelsize=16)
# plt.savefig('v2.png', dpi=200)
# plt.clf()
fig.subplots_adjust(top=0.92, left=0.05, right=0.99, bottom=0.25)

leg = axes[-2].legend(loc='lower center', prop={'size': 38}, ncol=2,
                      bbox_to_anchor=(-0.08, -0.38))
plt.subplots_adjust(wspace=0.24, hspace=0.3)
fig.align_labels()
plt.savefig("Maze.pdf")