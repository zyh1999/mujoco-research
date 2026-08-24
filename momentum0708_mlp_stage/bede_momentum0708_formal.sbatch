#!/bin/bash
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
#SBATCH --job-name=mj-m0708
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.err

set -Eeuo pipefail
ROOT=/nobackup/projects/bdman37/yihe
REPO=$ROOT/src/ICML2026-RAT
STAGE=$REPO/momentum0708_mlp_stage
PY=$ROOT/ppc64le/envs/rat/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be exported}
ENVS=(halfcheetah humanoid humanoidstandup)
MOMENTA=(0.7 0.8)
ENV_NAME=${ENVS[$((SLURM_ARRAY_TASK_ID % 3))]}
MOMENTUM=${MOMENTA[$((SLURM_ARRAY_TASK_ID / 3))]}

module load gcc/12.2 cmake/3.18.4 cuda/12.4.1
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=$ROOT/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export LD_LIBRARY_PATH=$ROOT/local/glfw-conda/lib:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p "$TMPDIR"

"$PY" -u "$STAGE/run_full_momentum_cell.py" \
  --repo "$REPO" \
  --trainer "$STAGE/train_detach_smallbatch_momentum.py" \
  --config rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml \
  --python "$PY" \
  --run-root "$RUN_ROOT" \
  --env "$ENV_NAME" \
  --momentum "$MOMENTUM" \
  --gpu 0 \
  --seeds 0 1
