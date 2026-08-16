#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yihe/rat_dualenergyfree255p1_20260806
REPO="$ROOT/repo"
PY="$ROOT/conda/bin/python"
RUN_ROOT="$ROOT/perf_runs/workstation15_detach_dualenergyfree255p1_independentanchors_d003_kfalse_all7_5seed_10m_gpu1"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl

mkdir -p "$RUN_ROOT"
cp "$ROOT/train_detach_dualenergyfree255p1.py" "$RUN_ROOT/"
cp "$ROOT/run_detach_dualenergyfree_worker.py" "$RUN_ROOT/"
cp "$ROOT/run_info_detach.txt" "$RUN_ROOT/run_info.txt"
sha256sum \
  "$RUN_ROOT/train_detach_dualenergyfree255p1.py" \
  "$RUN_ROOT/run_detach_dualenergyfree_worker.py" \
  "$RUN_ROOT/run_info.txt" > "$RUN_ROOT/manifest.sha256"
printf '%s\n' "RUNNING" > "$RUN_ROOT/status"
for worker in 0 1; do
  "$PY" -u "$RUN_ROOT/run_detach_dualenergyfree_worker.py" \
    --run-root "$RUN_ROOT" \
    --worker-index "$worker" \
    --python "$PY" \
    --repo "$REPO" \
    --max-concurrent-seeds 2
done
printf '%s\n' "FINISHED" > "$RUN_ROOT/status"
