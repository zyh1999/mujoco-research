#!/bin/bash --login
#SBATCH --job-name=mj-m0708
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-5%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.err

set -Eeuo pipefail
REPO=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
STAGE=$REPO/momentum0708_mlp_stage_20260824
PY=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be exported}
ENVS=(halfcheetah humanoid humanoidstandup)
MOMENTA=(0.7 0.8)
ENV_NAME=${ENVS[$((SLURM_ARRAY_TASK_ID % 3))]}
MOMENTUM=${MOMENTA[$((SLURM_ARRAY_TASK_ID / 3))]}
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl

"$PY" -u "$STAGE/run_full_momentum_cell.py" \
  --repo "$REPO" --trainer "$STAGE/train_detach_smallbatch_momentum.py" \
  --config rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml \
  --python "$PY" --run-root "$RUN_ROOT" --env "$ENV_NAME" \
  --momentum "$MOMENTUM" --gpu 0 --seeds 0 1
