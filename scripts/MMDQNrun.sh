rm -rf ./dopamine_runs/$1_MMDQN$4
CUDA_VISIBLE_DEVICES=${2:-0} python -um dopamine.discrete_domains.train \
    --base_dir ./dopamine_runs/$1_MMDQN$4 \
    --gin_files dopamine/agents/mmdqn/configs/mmdqn.gin \
    --gin_bindings atari_lib.create_atari_environment.game_name=\"$1\" \
    --gin_bindings Runner.reward_logdir=\"./reward-compose/$1-reward.txt\" \
    $3