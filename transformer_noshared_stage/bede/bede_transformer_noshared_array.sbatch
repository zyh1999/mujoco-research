#!/bin/bash
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-6
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.err

set -Eeuo pipefail

ROOT=/nobackup/projects/bdman37/yihe
ENVS=(ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d)
export ENV_NAME=${ENVS[$SLURM_ARRAY_TASK_ID]}
export RUN_ROOT=$ROOT/perf_runs/bede_transformer_noshared_${METHOD}_all7_5seed_10m_e4x8_${SLURM_ARRAY_JOB_ID}
export TOTAL_TIMESTEPS=10000000
export SEED_LIST="0 1 2 3 4"

bash "$ROOT/transformer_noshared_20260813/bede_transformer_noshared_worker.sh"
