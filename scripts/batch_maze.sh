SAMPLE_NUM=200
EXTRA=" --gin_bindings mmdqn_nd_ev_agent.MMDAgent.num_atoms=${SAMPLE_NUM} --gin_bindings Runner.monte_carlo_samples=${SAMPLE_NUM}"
GPU=${1:-0}
nohup ./scripts/MMDQNNDEVrun.sh MultiRewardMaze-v0 $GPU "$EXTRA" "_network-v21" > logs/Maze-v0-MMDQNND-network-v21.txt 2>&1 &
nohup ./scripts/MMDQNNDEVrun.sh MultiRewardMaze-v1 $GPU "$EXTRA" "_network-v21" > logs/Maze-v1-MMDQNND-network-v21.txt 2>&1 &
nohup ./scripts/MMDQNNDEVrun.sh MultiRewardMaze-v2 $GPU "$EXTRA" "_network-v21" > logs/Maze-v2-MMDQNND-network-v21.txt 2>&1 &