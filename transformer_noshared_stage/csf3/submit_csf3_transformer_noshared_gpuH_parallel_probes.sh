#!/bin/bash
set -Eeuo pipefail

SOURCE=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1
SBATCH=$SOURCE/csf3/csf3_transformer_noshared_gpuH_parallel_probe.sbatch
TIMESTEPS=${TOTAL_TIMESTEPS:-163840}

printf "phase\tjob_id\tparallelism\ttasks\ttimesteps_per_child\n" > "$SOURCE/csf3/submitted_gpuH_parallel_probes.tsv"
for spec in "p5 5 5" "p8 8 8" "p16a 16 16" "p16b 16 16"; do
  read -r tag parallel tasks <<< "$spec"
  out=$(sbatch --parsable --export=ALL,PROBE_TAG="$tag",NUM_GPUS=1,MAX_PARALLEL_PER_GPU="$parallel",TASK_LIMIT="$tasks",TOTAL_TIMESTEPS="$TIMESTEPS" "$SBATCH")
  job_id=${out%%;*}
  printf "%s\t%s\t%s\t%s\t%s\n" "$tag" "$job_id" "$parallel" "$tasks" "$TIMESTEPS" \
    >> "$SOURCE/csf3/submitted_gpuH_parallel_probes.tsv"
done
cat "$SOURCE/csf3/submitted_gpuH_parallel_probes.tsv"
