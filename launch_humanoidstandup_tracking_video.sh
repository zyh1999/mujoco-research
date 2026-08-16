#!/bin/bash
#SBATCH -J humstand_video_track
#SBATCH -p interactive
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00

set -euo pipefail

BASE=/scratch/h99859yz/rat_default_shared_csf3/scratch/humanoidstandup_checkpoint_compare_20260803/final_high_287936_vs_ordinary_153497
OUT=${BASE}/rendered_tracking_sample_seed0_egl
PYTHON=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
REPO=/scratch/h99859yz/trust-region-main

export MUJOCO_GL=egl
mkdir -p "${OUT}"
"${PYTHON}" "${BASE}/record_humanoidstandup_checkpoint_compare_tracking.py" \
  --repo "${REPO}" \
  --high-checkpoint "${BASE}/high_287936_seed0.ckpt" \
  --ordinary-checkpoint "${BASE}/ordinary_153497.ckpt" \
  --output-dir "${OUT}" \
  --seed 0 \
  --episodes 1 \
  --steps 1000 \
  --fps 30 \
  --device cpu \
  --action-mode sample \
  --action-seed 0 \
  --backend direct
