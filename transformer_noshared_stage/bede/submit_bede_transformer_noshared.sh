#!/bin/bash
set -Eeuo pipefail

ROOT=/nobackup/projects/bdman37/yihe
LAUNCH_ROOT=$ROOT/transformer_noshared_20260813
LOG_ROOT=$ROOT/logs
mkdir -p "$LOG_ROOT"

smoke_job=$(sbatch --parsable "$LAUNCH_ROOT/bede_transformer_noshared_smoke.sbatch")
smoke_job=${smoke_job%%;*}
printf "smoke\t%s\n" "$smoke_job" > "$LAUNCH_ROOT/submitted_jobs.tsv"

submit_method() {
  local method=$1
  local job_name=$2
  local job_id
  job_id=$(sbatch --parsable \
    --dependency="afterok:$smoke_job" \
    --job-name="$job_name" \
    --export="ALL,METHOD=$method" \
    "$LAUNCH_ROOT/bede_transformer_noshared_array.sbatch")
  job_id=${job_id%%;*}
  printf "%s\t%s\n" "$method" "$job_id" >> "$LAUNCH_ROOT/submitted_jobs.tsv"
}

submit_method ppo tr_nsh_ppo
submit_method kfac tr_nsh_kfac
submit_method emp256 tr_nsh_e256
submit_method energyfree255p1 tr_nsh_ef255
submit_method fullemp tr_nsh_full

cat "$LAUNCH_ROOT/submitted_jobs.tsv"
