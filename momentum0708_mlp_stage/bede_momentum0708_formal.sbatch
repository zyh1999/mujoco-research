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
ENVS=(ant halfcheetah hopper humanoid humanoidstandup walker2d)
ENV_NAME=${ENVS[$SLURM_ARRAY_TASK_ID]}

module load gcc/12.2 cmake/3.18.4 cuda/12.4.1
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=$ROOT/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export LD_LIBRARY_PATH=$ROOT/local/glfw-conda/lib:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p "$TMPDIR"

run_momentum() {
  local momentum=$1
  "$PY" -u "$STAGE/run_full_momentum_cell.py" \
    --repo "$REPO" \
    --trainer "$STAGE/train_detach_smallbatch_momentum.py" \
    --config rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml \
    --python "$PY" \
    --run-root "$RUN_ROOT" \
    --env "$ENV_NAME" \
    --momentum "$momentum" \
    --gpu 0 \
    --parallel-seeds \
    --seeds 0 1
}

run_momentum 0.7 > "$RUN_ROOT/formal_${ENV_NAME}_m07.launch.log" 2>&1 & p0=$!
run_momentum 0.8 > "$RUN_ROOT/formal_${ENV_NAME}_m08.launch.log" 2>&1 & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
exit "$rc"
