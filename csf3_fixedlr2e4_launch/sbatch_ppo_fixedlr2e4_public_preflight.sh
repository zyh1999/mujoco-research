#!/bin/bash --login
#SBATCH -J ppo2e4_pf
#SBATCH -p gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH -o /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%j.out
#SBATCH -e /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%j.err
#SBATCH -D /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original

set -euo pipefail
ROOT=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
PY=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_shared_ppo_public.py \
  --config ppo_mlp_shared_fixedlr2e4_public.yaml \
  --env_name halfcheetah --seed 99 --device 0 --timesteps_per_proc 8192
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_detach_ppo_public.py \
  --config ppo_mlp_detach_fixedlr2e4_public.yaml \
  --env_name halfcheetah --seed 99 --device 0 --timesteps_per_proc 8192
