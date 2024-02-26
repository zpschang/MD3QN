# Distributional Reinforcement Learning for Multi-Dimensional Reward Functions

Implementations for our paper [*Distributional Reinforcement Learning for Multi-Dimensional Reward Functions*](https://papers.nips.cc/paper_files/paper/2021/hash/0b9e57c46de934cee33b0e8d1839bfc2-Abstract.html) at NeurIPS 2021.

## Codes

The implementation of three maze environments is in `dopamine/environment/maze.py`,
and the implementation of MD3QN is in `dopamine/agents/mmdqn_nd/mmdqn_nd_agent.py`. 

## Dependencies

Run 

```
conda env create -f environment.yml
```

in this repository to install all dependencies for this project. 

## Run experiments

Run `python ./script/batch_general.py` to run the experiments for both policy evaluation setting and control setting. 

`batch_general.py` provides an interactive process for user to input the configurations, and run the experiment on these configurations.

By running `python ./script/batch_general.py` it runs all the experiment with the input configurations.

By running `python ./script/batch_general.py export`, it exports all the commands for experiments (without running) into `./scripts-export`. 
By running `python ./script/batch_general.py test`, it outputs the command for experiments (without running).

### Modeled Joint distribution in policy evaluation on Maze environments
To reproduce Figure 2 in our paper for modeled joint distribution on policy evaluation setting on Maze,
run `batch_general.py` in the following configurations: 

* clip_reward: False
* icml_setting: False 
* game_name_file: maze.txt
* alg_name: MMDQNND
* seed: 0
* network_type: v21
* bandwidth_type: v3
* kscale_type: v11
* use_priority: False
* evaluation_setting: True
* eval_policy_path: use default value (press Enter)
* iterations: 20
* exp_name: use default value (press Enter)
* resume: False
* gpu_func: use default value (press Enter)

The results will be saved in `dopamine_runs/MultiRewardMaze-v0_MMDQNND_seed-0_network-v21_bw-v3_kscale-v11_eval/evaluation_plots/iter-{iteration}/episode-{episode}.pickle`

You can use the script `plot-maze.py` to plot the results in Figure 2.

### Control setting on Atari games

Run `python ./script/batch_general.py` with the following configurations to run the policy optimization experiments on Atari games. To generate the data needed for Figure 3:

* reward_clipping: True
* icml_setting: True
* game_name_file: default value (press Enter)
* alg_name: DQN or HRA or MMDQN or MMDQNND
* seed: 0 or 1 or 2
* if running MD3QN:
    * network_type: v21
    * bandwidth_type: v3
    * kscale_type: v11
    * use_priority: False
    * evaluation_setting: False
* iterations: 200
* exp_name: use default value (press Enter)
* resume: False
* gpu_func: use default value (press Enter)

The results will be saved in `dopamine_runs/{game_name}_MMDQNND_icml_seed-{seed}_network-v21_bw-v3_kscale-v11`
and `{game_name}_HRA_icml_seed-2`

We provide `plot-atari.py` to generate the Figure 3 in our paper. You can copy all the results for HRA, MMDQN and MD3QN in `dopamine_runs/` into `data/` folder, and run `python plot-atari.py`, which will generate Figure 3 in our paper. 

## RL with multiple constraints by MD3QN

To reproduce the experiments for Appendix A.3.3 (RL with multiple constraints), run `python ./script/batch_general.py` with the following configurations: 

* clip_reward: False
* icml_setting: False 
* game_name_file: maze-v3.txt
* alg_name: MMDQNND
* seed: 0
* network_type: v21
* bandwidth_type: v3
* kscale_type: v11
* use_priority: False
* evaluation_setting: False
* eval_policy_path: use default value (press Enter)
* iterations: 20
* exp_name: use default value (press Enter)
* resume: False
* gpu_func: use default value (press Enter)
* First use_marginal: False (using MD3QN's joint distribution)
* Second use_marginal: True (using MD3QN's marginal distribution information)

