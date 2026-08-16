#!/bin/bash --login
#SBATCH -J detppo_2e4
#SBATCH -p gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH --array=0-6%2
#SBATCH -o /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.out
#SBATCH -e /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.err
#SBATCH -D /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original

set -euo pipefail
ROOT=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
ENVS=(ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d)
export MODE=detach
export ENV_NAME=${ENVS[$SLURM_ARRAY_TASK_ID]}
export RUN_ROOT="$ROOT/perf_runs/csf3_detach_ppo_fixedlr2e4_public_all7_5seed_10m_e4x8_gpuL_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$ROOT/slurm_logs" "$RUN_ROOT"
bash "$ROOT/launch_public_ppo_fixedlr2e4_env.sh"
