#!/usr/bin/env bash
#SBATCH --job-name=mj-ef255p1-lbs-l2
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-6%2
#SBATCH --output=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x-%A_%a.out
#SBATCH --error=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x-%A_%a.err
#SBATCH --chdir=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original

set -euo pipefail

REPO=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
PY=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
RUN_ROOT="$REPO/perf_runs/csf3_detjc_largebs_energyfree255p1_ggn256_l2clip05_d003_kfalse_all7_5seed_500u_gpuL_20260810"
ENVS=(ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d)
ENV_NAME=${ENVS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$RUN_ROOT" "$REPO/slurm_logs"
printf '%s\n' "$SLURM_JOB_ID" > "$RUN_ROOT/job_${ENV_NAME}.txt"

export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PY" -u "$REPO/run_energyfree255p1_batch262144_worker.py" \
  --run-root "$RUN_ROOT" \
  --repo "$REPO" \
  --python "$PY" \
  --variant l2 \
  --device 0 \
  --envs "$ENV_NAME" \
  --max-concurrent 5
