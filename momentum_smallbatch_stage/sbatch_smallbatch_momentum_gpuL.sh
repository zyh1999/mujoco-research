#!/bin/bash --login
#SBATCH --job-name=mj-sb-momentum
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-41%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.err
#SBATCH --chdir=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original

set -euo pipefail
REPO=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
PY=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be exported with sbatch --export}
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PY" -u "$RUN_ROOT/run_smallbatch_momentum_worker.py" \
  --run-root "$RUN_ROOT" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --python "$PY" \
  --repo "$REPO"
