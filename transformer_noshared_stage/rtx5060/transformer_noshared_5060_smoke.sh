#!/usr/bin/env bash
set -Eeuo pipefail

STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack5060}
SOURCE_NAME=${SOURCE_NAME:-transformer_noshared_20260813_v1}
RUN_ROOT=${RUN_ROOT:-perf_runs/transformer_noshared_5060_smoke_$(date +%Y%m%d_%H%M%S)}
LAUNCH_ROOT=$STACK_ROOT/workspaces/transformer_noshared_5060_launch
HOST_SOURCE=$STACK_ROOT/workspaces/$SOURCE_NAME
HOST_RUN_ROOT=$STACK_ROOT/workspaces/$RUN_ROOT
mkdir -p "$HOST_RUN_ROOT"

echo RUNNING > "$HOST_RUN_ROOT/status"
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu --format=csv > "$HOST_RUN_ROOT/gpu_start.csv"

"$STACK_ROOT/bin/mujoco_container.sh" rat 0 bash -lc "
  set -Eeuo pipefail
  cd '/workspace/user/$SOURCE_NAME'
  python -m py_compile \
    utils/mujoco_transformer.py \
    kfac/kfac.py \
    kfac/kfac_utils.py \
    train_detach_ppo_public.py \
    train_detach_jointcritic_actor_fvp_fisherclip.py \
    train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py \
    train_detach_energyfree255p1_criticggn256_batch262144.py
  python verify_transformer_noshared.py
" > "$HOST_RUN_ROOT/unit.stdout" 2> "$HOST_RUN_ROOT/unit.stderr"
grep -q TRANSFORMER_NOSHARED_VERIFY_OK "$HOST_RUN_ROOT/unit.stdout"

smoke_one() {
  local method=$1
  local gpu=$2
  local seed=$3
  ALLOW_EMPTY_PROGRESS=1 TOTAL_TIMESTEPS=0 "$LAUNCH_ROOT/transformer_noshared_5060_run_seed.sh" \
    "$method" halfcheetah "$seed" "$gpu" "$RUN_ROOT" "$SOURCE_NAME"
}

smoke_one ppo 0 0
smoke_one kfac 1 0
smoke_one emp256 0 0
smoke_one energyfree255p1 1 0
smoke_one fullemp 0 0

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu --format=csv > "$HOST_RUN_ROOT/gpu_end.csv"
echo FINISHED > "$HOST_RUN_ROOT/status"
printf "%s\n" "$RUN_ROOT" > "$STACK_ROOT/workspaces/transformer_noshared_5060_launch/latest_smoke_root.txt"
