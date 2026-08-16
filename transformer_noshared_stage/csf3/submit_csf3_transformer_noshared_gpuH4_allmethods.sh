#!/bin/bash
set -Eeuo pipefail

SOURCE=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1
out=$(sbatch --parsable --export=ALL,NUM_GPUS=3,MAX_PARALLEL_PER_GPU=5 "$SOURCE/csf3/csf3_transformer_noshared_gpuH4_allmethods.sbatch")
job_id=${out%%;*}
run_root="$SOURCE/perf_runs/csf3_transformer_noshared_allmethods_all7_5seed_10m_e4x8_gpuH3_${job_id}"
printf "job_id\tpartition\tgpus\tparallel_per_gpu\trun_root\n%s\tgpuH\t3\t5\t%s\n" "$job_id" "$run_root" > "$SOURCE/csf3/submitted_gpuH4_allmethods.tsv"
echo "submitted job=$job_id run_root=$run_root"
