import os
import sys

mode = 'run'  # run the scripts for the input configurations
if sys.argv[-1] == 'test':  # only print the scripts for the input configurations
    mode = 'test'
elif sys.argv[-1] == 'export':  # export the scripts for the input configurations
    mode = 'export'
print(f'==== In {mode} mode ====')

def get_input(prompt, defaults):
    while True:
        x = input(f'{prompt} ({defaults}): ')
        if not x:
            return defaults[0]
        try:
            obj = eval(x)
        except:
            obj = x
        if obj in defaults or ... in defaults:
            return obj

extra_command = ''
suffix = ''
clip_reward = get_input('clip_reward', [True, False])  # whether clip reward to 1.0. set True in our experiments
if not clip_reward:
    extra_command += ' --gin_bindings Runner.clip_rewards=False'
    suffix += '_noclip'

icml_setting = get_input('icml_setting', [True, False])  # whether use ICML settings. set True in our experiments
if icml_setting:
    extra_command += ' --gin_bindings atari_lib.create_atari_environment.sticky_actions=False'
    extra_command += ' --gin_bindings AtariPreprocessing.terminal_on_life_loss=True'
    suffix += '_icml'


game_name_file = get_input('game_name_file', ['atari.txt', 'maze.txt', 'maze-v3.txt', ...])
alg_name = get_input('alg_name', ['DQN', 'HRA', 'IQN', 'MMDQN', 'MMDQNND'])

seed = get_input('seed', [0, 1, 2, 3, ...])
suffix += f'_seed-{seed}'

if alg_name == 'MMDQNND':
    # network architecture to use. In our experiment, set to v21.
    network_type = get_input('network_type', ['v1', 'v12', 'v21', 'v22', 'v3-drdrl'])
    extra_command += f' --gin_bindings mmdqn_nd_agent.MMDAgent.network_type=\'{network_type}\''
    suffix += f'_network-{network_type}'
    if network_type == 'v1' or network_type == 'v12':
        extra_command += ' --gin_bindings mmdqn_nd_agent.MMDAgent.num_atoms=100'
    if network_type == 'v22':
        extra_command += ' --gin_bindings mmdqn_nd_agent.MMDAgent.num_atoms=32'

    # bandwidth set to use. In our experiment, set to v3.
    bandwidth_type = get_input('bandwidth_type', ['v0', 'v1', 'v22', 'v3', 'v4', 'v5'])
    extra_command += f' --gin_bindings mmdqn_nd_agent.MMDAgent.bandwidth_type=\'{bandwidth_type}\''
    suffix += f'_bw-{bandwidth_type}'

    # a trick to add weights to gaussian kernels according to bandwidths on the kernel function.
    # In our experiment, set to v11.
    kscale_type = get_input('kscale_type', ['v0', 'v11', 'v12'])
    extra_command += f' --gin_bindings mmdqn_nd_agent.MMDAgent.kscale_type=\'{kscale_type}\''
    suffix += f'_kscale-{kscale_type}'

    # whether to use prioritized experience replay. In our experiment, set to False.
    use_priority = get_input('use_priority', [False, True])
    if use_priority:
        extra_command += f' --gin_bindings mmdqn_nd_agent.MMDAgent.use_priority=True'
        suffix += '_prior'

    # False: policy evaluation setting; True: control setting.
    evaluation_setting = get_input('evaluation_setting', [False, True])
    if evaluation_setting:
        extra_command += f' --gin_bindings mmdqn_nd_agent.MMDAgent.evaluation_setting=True'
        suffix += '_eval'
        eval_policy_path = get_input('eval_policy_path', ['eval_policy/%s_IQN_icml_seed-0', ...])

iterations = get_input('iterations', [200, ...])
extra_command += f' --gin_bindings Runner.num_iterations={iterations}'

exp_name = get_input('exp_name', ['MD3QN-initial-721', ...])
extra_command += f' --gin_bindings create_runner.exp_name=\'{exp_name}\''

resume = get_input('resume (only supported by MMDQNND)', [False, True])
resume_cmd = 'true' if resume else 'false'

gpu_func_str = get_input('gpu_func', ['i % 4', ...])
gpu_func = lambda i: (eval(gpu_func_str) if type(gpu_func_str) is str else gpu_func_str)

with open(game_name_file, 'r') as file:
    game_name_input = file.read()
game_names = game_name_input.strip().split(' ')
print(game_names)

if mode == 'export':
    idx = get_input('idx', [0, 1, ...])

os.system('set -x')

for i, game_name in enumerate(game_names):
    print(f'game_name: {game_name}')
    extra_command_new = extra_command
    suffix_new = suffix
    if alg_name == 'MMDQNND':
        if 'MultiRewardMaze' in game_name:  # four maze environment, where MultiRewardMaze-v3 is RL with multiple constraints environment
            extra_command_new += f' --gin_bindings mmdqn_nd_agent.MMDAgent.maze_env=True'
        if game_name == 'MultiRewardMaze-v3':  # RL with multiple constraints environment
            extra_command_new += f' --gin_bindings mmdqn_nd_agent.MMDAgent.constrain_env=True'
            use_marginal = get_input('use_marginal', [False, True])
            if use_marginal:
                extra_command_new += f' --gin_bindings mmdqn_nd_agent.MMDAgent.use_marginal=True'
                suffix_new += f'_use_marginal'
        if evaluation_setting and 'MultiRewardMaze' not in game_name:
            eval_policy_path_env = eval_policy_path % game_name
            extra_command_new += f' --gin_bindings mmdqn_nd_agent.MMDAgent.eval_policy_path=\'{eval_policy_path_env}\''
    gpu_id = gpu_func(i)
    print('gpu_id', gpu_id)
    cmd = f'nohup ./scripts/{alg_name}run.sh {game_name} {gpu_id} "{extra_command_new}" "{suffix_new}" "{resume_cmd}" > logs/{game_name}-{alg_name}{suffix_new}.txt 2>&1 &'
    print(f'cmd: {cmd}')
    if mode == 'run':
        os.system(cmd)
    if mode == 'export':
        os.makedirs(f'scripts-export', exist_ok=True)
        with open(f'scripts-export/{idx}-{game_name}.sh', 'w') as file:
            file.write(f'echo Staring experiment "{exp_name}"\n')
            file.write(f'nohup ./scripts/{alg_name}run.sh {game_name} {gpu_id} "{extra_command_new}" "{suffix}" &\n')
