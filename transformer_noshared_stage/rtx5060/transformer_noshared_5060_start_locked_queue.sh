#!/usr/bin/env bash
set -Eeuo pipefail

STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack5060}
SOURCE_NAME=${SOURCE_NAME:-transformer_noshared_20260813_v1}
RUN_ROOT=${RUN_ROOT:-perf_runs/transformer_noshared_5060_locked_all_methods_all7_5seed_10m_$(date +%Y%m%d_%H%M%S)}
PER_GPU_WORKERS=${PER_GPU_WORKERS:-2}
LAUNCH_ROOT=$STACK_ROOT/workspaces/transformer_noshared_5060_launch
HOST_RUN_ROOT=$STACK_ROOT/workspaces/$RUN_ROOT
TASKS_TSV=$HOST_RUN_ROOT/tasks.tsv
mkdir -p "$HOST_RUN_ROOT/workers" "$STACK_ROOT/logs/mujoco/transformer_noshared"

methods=(ppo kfac emp256 energyfree255p1 fullemp)
envs=(ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d)

printf "shard\tmethod\tenv\tseed\n" > "$TASKS_TSV"
i=0
for method in "${methods[@]}"; do
  for env_name in "${envs[@]}"; do
    for seed in 0 1 2 3 4; do
      shard=$((i % 2))
      printf "%s\t%s\t%s\t%s\n" "$shard" "$method" "$env_name" "$seed" >> "$TASKS_TSV"
      i=$((i + 1))
    done
  done
done

{
  echo "source_name=$SOURCE_NAME"
  echo "run_root=$HOST_RUN_ROOT"
  echo "tasks=$TASKS_TSV"
  echo "methods=${methods[*]}"
  echo "envs=${envs[*]}"
  echo "seeds=0 1 2 3 4"
  echo "total_runs=$((i))"
  echo "per_gpu_workers=$PER_GPU_WORKERS"
  echo "worker_mode=locked_skip_running_finished"
  echo "started=$(date --iso-8601=seconds)"
} > "$HOST_RUN_ROOT/queue_info.txt"
echo RUNNING > "$HOST_RUN_ROOT/status"

for gpu in 0 1; do
  for worker_id in $(seq 0 $((PER_GPU_WORKERS - 1))); do
    log="$HOST_RUN_ROOT/workers/gpu${gpu}_${worker_id}.nohup.log"
    TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000} nohup \
      "$LAUNCH_ROOT/transformer_noshared_5060_worker_locked.sh" "$gpu" "$worker_id" "$TASKS_TSV" "$RUN_ROOT" "$SOURCE_NAME" \
      > "$log" 2>&1 &
    echo "$!" > "$HOST_RUN_ROOT/workers/gpu${gpu}_${worker_id}.hostpid"
  done
done

printf "%s\n" "$RUN_ROOT" > "$LAUNCH_ROOT/latest_locked_queue_root.txt"
cat "$HOST_RUN_ROOT/queue_info.txt"
