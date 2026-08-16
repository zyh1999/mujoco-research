#!/bin/bash --login
#SBATCH -J mj_d262k
#SBATCH -p gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-3%4
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=120G
#SBATCH -t 2-00:00:00
#SBATCH -o /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.out
#SBATCH -e /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.err
#SBATCH -D /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original

set -euo pipefail
ROOT=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
PY=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be exported}
SHARD_BASE=${SHARD_BASE:-0}
SHARD=$((SLURM_ARRAY_TASK_ID + SHARD_BASE))

export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PY" -u "$RUN_ROOT/run_batch262144_worker.py" \
  --run-root "$RUN_ROOT" \
  --shard "$SHARD" \
  --python "$PY" \
  --repo "$ROOT"
