#!/bin/bash --login
#SBATCH -J mj_e256_kt_pf
#SBATCH -p gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%j.out
#SBATCH -e /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%j.err
#SBATCH -D /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original

set -euo pipefail
ROOT=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
PY=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be exported}

export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for shard in 0 1; do
  "$PY" -u "$RUN_ROOT/run_batch262144_clipped_worker.py" \
    --run-root "$RUN_ROOT" \
    --shard "$shard" \
    --python "$PY" \
    --repo "$ROOT" \
    --smoke-only
done
