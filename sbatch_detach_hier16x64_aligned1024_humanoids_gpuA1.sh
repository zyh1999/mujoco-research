#!/bin/bash --login
#SBATCH -J ns_h16x64_humA
#SBATCH -p gpuA
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --array=0-1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=120G
#SBATCH -t 1-12:00:00
#SBATCH -o /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.out
#SBATCH -e /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.err
#SBATCH -D /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original

set -euo pipefail
ROOT=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
ENVS=(humanoid humanoidstandup)
ENV_NAME=${ENVS[$SLURM_ARRAY_TASK_ID]}
RUN_ROOT="$ROOT/perf_runs/csf3_detach_hier16x64_aligned1024_ggn_normnone_damp003_kfalse_${ENV_NAME}_5seed_10m_gpuA1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export RUN_ROOT ENV_NAME DAMPING=0.03
mkdir -p "$ROOT/slurm_logs" "$RUN_ROOT"
bash "$ROOT/launch_detach_hier16x64_aligned1024_env.sh"
