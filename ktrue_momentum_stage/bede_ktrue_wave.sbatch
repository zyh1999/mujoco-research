#!/usr/bin/env bash
#SBATCH --account=bdman37
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --job-name=mj-ktrue-wave
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.err

set -Eeuo pipefail
ROOT=/nobackup/projects/bdman37/yihe
REPO=$ROOT/src/ICML2026-RAT
PY=$ROOT/ppc64le/envs/rat/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be exported}
WAVE=${WAVE:?WAVE must be exported}
MOMENTA=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.8 0.8 0.8 0.8 0.8 0.8 0.8)
ENVS=(ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d)

module load gcc/12.2 cmake/3.18.4 cuda/12.4.1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TMPDIR=$ROOT/tmp/ktrue_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TMPDIR"

run_pair() {
  local pair=$1 momentum=${MOMENTA[$1]} env_name=${ENVS[$1]}
  "$PY" -u "$REPO/momentum0708_mlp_stage/run_full_momentum_cell.py" \
    --repo "$REPO" \
    --trainer "$REPO/momentum0708_mlp_stage/train_detach_smallbatch_momentum.py" \
    --config rat_mlp_detjc_normnone_ktrue_d003_lrv01_e4_mlp.yaml \
    --python "$PY" \
    --run-root "$RUN_ROOT" \
    --env "$env_name" \
    --momentum "$momentum" \
    --gpu 0 \
    --parallel-seeds \
    --seeds 0 1 \
    --require-kaczmarz
}

if [[ "$WAVE" == 1 ]]; then start=0; stop=12; else start=12; stop=21; fi
rc=0
pids=()
for ((pair=start + SLURM_ARRAY_TASK_ID; pair<stop; pair+=6)); do
  run_pair "$pair" > "$RUN_ROOT/wave${WAVE}_pair${pair}.launch.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit "$rc"
