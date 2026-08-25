#!/usr/bin/env bash
#SBATCH --account=bdman37
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --job-name=mj-ktrue-pf
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.err

set -Eeuo pipefail
ROOT=/nobackup/projects/bdman37/yihe
REPO=$ROOT/src/ICML2026-RAT
STAGE=$REPO/ktrue_momentum_stage
PY=$ROOT/ppc64le/envs/rat/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be exported}
MOMENTA=(0.5 0.7 0.8)
MOMENTUM=${MOMENTA[$SLURM_ARRAY_TASK_ID]}

module load gcc/12.2 cmake/3.18.4 cuda/12.4.1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TMPDIR=$ROOT/tmp/ktrue_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TMPDIR"

"$PY" -u "$REPO/momentum0708_mlp_stage/run_full_momentum_cell.py" \
  --repo "$REPO" \
  --trainer "$REPO/momentum0708_mlp_stage/train_detach_smallbatch_momentum.py" \
  --config rat_mlp_detjc_normnone_ktrue_d003_lrv01_e4_mlp.yaml \
  --python "$PY" \
  --run-root "$RUN_ROOT" \
  --env halfcheetah \
  --momentum "$MOMENTUM" \
  --gpu 0 \
  --seeds 0 \
  --require-kaczmarz \
  --preflight
