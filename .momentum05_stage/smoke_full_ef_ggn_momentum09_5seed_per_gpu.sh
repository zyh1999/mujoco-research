#!/bin/bash
set -euo pipefail

ROOT=/home/yihe/ICML2026-RAT-original-4090
PY=/home/yihe/.venv/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be set}
TRAINER=train_detach_jointcritic_momentum09.py
CONFIG=rat_mlp_detjc_full_ef_full_ggn_momentum09_1024x8_e4.yaml

cd "$ROOT"
mkdir -p "$RUN_ROOT/preflight"

export PATH=/home/yihe/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

command_for() {
  local env=$1 seed=$2 timesteps=$3
  printf '%q ' "$PY" -u "$TRAINER" \
    --config "$CONFIG" --env_name "$env" --seed "$seed" --device 0 \
    --timesteps_per_proc "$timesteps" --pi_epochs 4 --lr_pi 0.05 \
    --cg_damping 0.03 --fisher_kernel exact \
    --fisher_kernel_normalization none \
    --actor_curvature_subsample 0 --critic_curvature_subsample 0 \
    --critic_update gn --critic_optimizer sgd \
    --no-actor_subsample_full_batch_gradient
}

run_gpu_smoke() {
  local gpu=$1 env=$2 seed_base=$3
  local gpu_dir="$RUN_ROOT/preflight/gpu${gpu}"
  mkdir -p "$gpu_dir"
  local pids=() seeds=()

  for offset in 0 1 2 3 4; do
    local seed=$((seed_base + offset))
    local command
    command=$(command_for "$env" "$seed" 1)
    printf '%s\n' "$command" > "$gpu_dir/seed${seed}.command.txt"
    CUDA_VISIBLE_DEVICES=$gpu bash -lc "$command" \
      > "$gpu_dir/seed${seed}.stdout" 2> "$gpu_dir/seed${seed}.stderr" &
    pids+=("$!")
    seeds+=("$seed")
  done

  local peak_mib=0 active=1
  while [[ $active -eq 1 ]]; do
    active=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then active=1; fi
    done
    local used
    used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used > peak_mib )); then peak_mib=$used; fi
    [[ $active -eq 0 ]] || sleep 1
  done

  local failed=0
  for i in "${!pids[@]}"; do
    local rc=0
    wait "${pids[$i]}" || rc=$?
    printf '%s\n' "$rc" > "$gpu_dir/seed${seeds[$i]}.rc"
    [[ $rc -eq 0 ]] || failed=1
  done
  printf '%s\n' "$peak_mib" > "$gpu_dir/peak_gpu_memory_mib.txt"
  return "$failed"
}

echo RUNNING > "$RUN_ROOT/preflight/status"
run_gpu_smoke 0 halfcheetah 9000 & p0=$!
run_gpu_smoke 1 ant 9010 & p1=$!
rc0=0; rc1=0
wait "$p0" || rc0=$?
wait "$p1" || rc1=$?

bad=0
if grep -R -E -i "CUDA out of memory|\bnan\b|Traceback|LinAlgError" "$RUN_ROOT/preflight" --include='*.stderr' > "$RUN_ROOT/preflight/error_scan.txt"; then
  bad=1
fi
momentum_count=$(grep -R -h -c "Independent SGD momentum: actor=0.9 critic=0.9" "$RUN_ROOT/preflight" --include='*.stdout' | awk '{s+=$1} END {print s+0}')
full_rows_count=$(grep -R -h -c "Curvature rows per minibatch: actor=1024 critic=1024" "$RUN_ROOT/preflight" --include='*.stdout' | awk '{s+=$1} END {print s+0}')

cat > "$RUN_ROOT/preflight/summary.txt" <<EOF
gpu0_rc=$rc0
gpu1_rc=$rc1
error_scan_bad=$bad
momentum_marker_count=$momentum_count
full_1024_rows_marker_count=$full_rows_count
gpu0_peak_mib=$(cat "$RUN_ROOT/preflight/gpu0/peak_gpu_memory_mib.txt")
gpu1_peak_mib=$(cat "$RUN_ROOT/preflight/gpu1/peak_gpu_memory_mib.txt")
EOF

if [[ $rc0 -eq 0 && $rc1 -eq 0 && $bad -eq 0 && $momentum_count -eq 10 && $full_rows_count -eq 10 ]]; then
  echo FINISHED > "$RUN_ROOT/preflight/status"
else
  echo FAILED > "$RUN_ROOT/preflight/status"
  exit 1
fi
