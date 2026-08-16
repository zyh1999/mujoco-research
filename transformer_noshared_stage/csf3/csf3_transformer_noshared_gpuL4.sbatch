#!/bin/bash --login
#SBATCH -J tr_tnosh_gpul4
#SBATCH -p gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:4
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=240G
#SBATCH -t 2-00:00:00
#SBATCH -o /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1/slurm_logs/%x_%j.out
#SBATCH -e /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1/slurm_logs/%x_%j.err
#SBATCH -D /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1

set -Eeuo pipefail

SOURCE=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1
METHOD=${METHOD:?METHOD must be exported}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
RUN_ROOT=${RUN_ROOT:-$SOURCE/perf_runs/csf3_transformer_noshared_${METHOD}_all7_5seed_10m_e4x8_gpuL4_${SLURM_JOB_ID}}

mkdir -p "$SOURCE/slurm_logs" "$RUN_ROOT"
export SOURCE METHOD TOTAL_TIMESTEPS RUN_ROOT

exec "$SOURCE/csf3/csf3_transformer_noshared_gpuL4_worker.sh"
