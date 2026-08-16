#!/bin/bash --login
#SBATCH -J tr_tnosh_probe
#SBATCH -p gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 01:30:00
#SBATCH -o /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1/slurm_logs/%x_%j.out
#SBATCH -e /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1/slurm_logs/%x_%j.err
#SBATCH -D /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1

set -Eeuo pipefail

SOURCE=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1
PROBE_TAG=${PROBE_TAG:?PROBE_TAG must be set}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-163840}
NUM_GPUS=1
MAX_PARALLEL_PER_GPU=${MAX_PARALLEL_PER_GPU:?MAX_PARALLEL_PER_GPU must be set}
TASK_LIMIT=${TASK_LIMIT:?TASK_LIMIT must be set}
RUN_ROOT=${RUN_ROOT:-$SOURCE/perf_runs/csf3_transformer_noshared_parallel_probe_${PROBE_TAG}_${SLURM_JOB_ID}}

mkdir -p "$SOURCE/slurm_logs" "$RUN_ROOT"
export SOURCE TOTAL_TIMESTEPS NUM_GPUS MAX_PARALLEL_PER_GPU TASK_LIMIT RUN_ROOT

{
  echo "probe_tag=$PROBE_TAG"
  echo "requested_parallelism=$MAX_PARALLEL_PER_GPU"
  echo "task_limit=$TASK_LIMIT"
  echo "timesteps_per_child=$TOTAL_TIMESTEPS"
  echo "gpu_allocation=h200:1"
  echo "cpu_allocation=8"
  echo "telemetry_interval_seconds=5"
} > "$RUN_ROOT/probe_info.txt"

collect_gpu() {
  echo "timestamp_utc,index,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w" > "$RUN_ROOT/gpu_telemetry.csv"
  while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits \
      | awk -v now="$(date -u +%FT%TZ)" -F ', ' '{print now "," $1 "," $2 "," $3 "," $4 "," $5}' \
      >> "$RUN_ROOT/gpu_telemetry.csv" || true
    sleep 5
  done
}

collect_gpu &
TELEMETRY_PID=$!
cleanup() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
}
trap cleanup EXIT

"$SOURCE/csf3/csf3_transformer_noshared_gpuH4_allmethods_worker.sh"
