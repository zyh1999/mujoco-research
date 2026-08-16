#!/usr/bin/env bash
set -euo pipefail

REPO=/home/yihe/ICML2026-RAT-original-4090
PY=/home/yihe/.venv/bin/python
RUN_ROOT="$REPO/perf_runs/workstation92_detjc_batch262144_curv256ggn_kopt_clip_momentum_optuna_2seed_20260805"
TRAINER="$REPO/train_detach_jointcritic_curv256_ktrue_clip_momentum_optuna.py"
WORKER="$REPO/run_curv256_ktrue_clip_momentum_optuna_worker.py"
CONFIG=rat_mlp_detjc_curv256_criticggn_ktrue_clip_momentum_optuna_batch262144_ws92.yaml

mkdir -p "$RUN_ROOT"
printf 'RUNNING\n' > "$RUN_ROOT/status"

export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 nohup "$PY" -u "$WORKER" \
  --gpu 0 \
  --envs ant humanoid \
  --repo "$REPO" \
  --root "$RUN_ROOT" \
  --trainer "$TRAINER" \
  --python "$PY" \
  --config "$CONFIG" \
  --new-trials 8 \
  --min-free-mb 6500 \
  > "$RUN_ROOT/worker_gpu0.stdout" 2> "$RUN_ROOT/worker_gpu0.stderr" < /dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/worker_gpu0.pid"

CUDA_VISIBLE_DEVICES=1 nohup "$PY" -u "$WORKER" \
  --gpu 1 \
  --envs halfcheetah walker2d \
  --repo "$REPO" \
  --root "$RUN_ROOT" \
  --trainer "$TRAINER" \
  --python "$PY" \
  --config "$CONFIG" \
  --new-trials 8 \
  --min-free-mb 6500 \
  > "$RUN_ROOT/worker_gpu1.stdout" 2> "$RUN_ROOT/worker_gpu1.stderr" < /dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/worker_gpu1.pid"
