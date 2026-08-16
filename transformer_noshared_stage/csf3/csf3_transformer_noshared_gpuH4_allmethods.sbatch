#!/bin/bash --login
#SBATCH -J tr_tnosh_h200x3
#SBATCH -p gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:3
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=360G
#SBATCH -t 2-00:00:00
#SBATCH -o /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1/slurm_logs/%x_%j.out
#SBATCH -e /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1/slurm_logs/%x_%j.err
#SBATCH -D /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1

set -Eeuo pipefail

SOURCE=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
NUM_GPUS=${NUM_GPUS:-3}
MAX_PARALLEL_PER_GPU=${MAX_PARALLEL_PER_GPU:-5}
RUN_ROOT=${RUN_ROOT:-$SOURCE/perf_runs/csf3_transformer_noshared_allmethods_all7_5seed_10m_e4x8_gpuH3_${SLURM_JOB_ID}}

mkdir -p "$SOURCE/slurm_logs" "$RUN_ROOT"
export SOURCE TOTAL_TIMESTEPS NUM_GPUS MAX_PARALLEL_PER_GPU RUN_ROOT

exec "$SOURCE/csf3/csf3_transformer_noshared_gpuH4_allmethods_worker.sh"
