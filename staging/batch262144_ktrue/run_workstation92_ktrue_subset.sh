#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yihe/ICML2026-RAT-original-4090
PY=/home/yihe/.venv/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be exported}
PHASE=${1:?phase must be preflight or formal}

export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$RUN_ROOT"

if [[ "$PHASE" == preflight ]]; then
  printf 'PREFLIGHT\n' > "$RUN_ROOT/status"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u "$RUN_ROOT/run_batch262144_ktrue_workstation_worker.py" \
    --run-root "$RUN_ROOT" --shard 0 --python "$PY" --repo "$ROOT" --smoke-only
  CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN_ROOT/run_batch262144_ktrue_workstation_worker.py" \
    --run-root "$RUN_ROOT" --shard 1 --python "$PY" --repo "$ROOT" --smoke-only
  printf 'PREFLIGHT_OK\n' > "$RUN_ROOT/status"
  exit 0
fi

if [[ "$PHASE" != formal ]]; then
  printf 'unknown phase: %s\n' "$PHASE" >&2
  exit 2
fi

printf 'RUNNING\n' > "$RUN_ROOT/status"

run_gpu() {
  local gpu=$1
  shift
  local task
  for task in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$RUN_ROOT/run_batch262144_ktrue_workstation_pack5.py" \
      --run-root "$RUN_ROOT" --task-index "$task" --python "$PY" --repo "$ROOT"
  done
}

# GPU0: L2 Ant then Humanoid. GPU1: FVP Ant then Humanoid.
run_gpu 0 0 6 &
pid0=$!
run_gpu 1 1 7 &
pid1=$!
printf '%s\n' "$pid0" > "$RUN_ROOT/gpu0_pipeline.pid"
printf '%s\n' "$pid1" > "$RUN_ROOT/gpu1_pipeline.pid"

rc=0
wait "$pid0" || rc=$?
wait "$pid1" || rc=$?
printf '%s\n' "$rc" > "$RUN_ROOT/rc"
if [[ $rc -eq 0 ]]; then
  printf 'FINISHED\n' > "$RUN_ROOT/status"
else
  printf 'FAILED\n' > "$RUN_ROOT/status"
fi
exit "$rc"
