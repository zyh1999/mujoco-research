#!/bin/bash
set -Eeuo pipefail

SOURCE=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1
METHODS=(ppo kfac emp256 energyfree255p1 fullemp)
mkdir -p "$SOURCE/slurm_logs"
printf "method\tjob_id\tsbatch\trun_root_pattern\n" > "$SOURCE/csf3/submitted_gpuL4_jobs.tsv"

for method in "${METHODS[@]}"; do
  out=$(sbatch --parsable --export=ALL,METHOD="$method" "$SOURCE/csf3/csf3_transformer_noshared_gpuL4.sbatch")
  job_id=${out%%;*}
  run_root="$SOURCE/perf_runs/csf3_transformer_noshared_${method}_all7_5seed_10m_e4x8_gpuL4_${job_id}"
  printf "%s\t%s\t%s\t%s\n" "$method" "$job_id" "$SOURCE/csf3/csf3_transformer_noshared_gpuL4.sbatch" "$run_root" \
    >> "$SOURCE/csf3/submitted_gpuL4_jobs.tsv"
  echo "submitted method=$method job=$job_id run_root=$run_root"
done
