rm -rf ./dopamine_runs/$1_prerun$4
CUDA_VISIBLE_DEVICES=${2:-0} python -um dopamine.discrete_domains.train \
    --base_dir ./dopamine_runs/$1_prerun$4 \
    --gin_files dopamine/agents/implicit_quantile/configs/implicit_quantile.gin \
    --gin_bindings atari_lib.create_atari_environment.game_name=\"$1\" \
    --gin_bindings Runner.prerun=True \
    --gin_bindings Runner.reward_logdir=\"./reward-compose/$1-reward.txt\" \
    --gin_bindings Runner.num_iterations=200 \
    $3
