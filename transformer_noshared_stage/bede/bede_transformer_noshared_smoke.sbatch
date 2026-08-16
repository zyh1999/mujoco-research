#!/bin/bash
#SBATCH --job-name=tr_nosh_smoke
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%j.err

set -Eeuo pipefail

ROOT=/nobackup/projects/bdman37/yihe
SOURCE=$ROOT/src/ICML2026-RAT-transformer-noshared-20260813-v1
CONDA_ROOT=$ROOT/ppc64le/miniconda
CONDA_ENV=$ROOT/ppc64le/envs/rat
GLFW_PREFIX=$ROOT/local/glfw-conda
SMOKE_ROOT=$ROOT/perf_runs/bede_transformer_noshared_smoke_${SLURM_JOB_ID}

module load gcc/12.2
module load cmake/3.18.4
module load cuda/12.4.1
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR=$ROOT/tmp/$SLURM_JOB_ID
export LD_LIBRARY_PATH=$GLFW_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PIP_CACHE_DIR=$ROOT/ppc64le/pip-cache
export XDG_CACHE_HOME=$ROOT/ppc64le/xdg-cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$TMPDIR" "$ROOT/logs" "$SMOKE_ROOT"

echo RUNNING > "$SMOKE_ROOT/status"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu --format=csv > "$SMOKE_ROOT/gpu_start.csv"
python -m py_compile \
  "$SOURCE/utils/mujoco_transformer.py" \
  "$SOURCE/kfac/kfac.py" \
  "$SOURCE/kfac/kfac_utils.py" \
  "$SOURCE/train_detach_ppo_public.py" \
  "$SOURCE/train_detach_jointcritic_actor_fvp_fisherclip.py" \
  "$SOURCE/train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py" \
  "$SOURCE/train_detach_energyfree255p1_criticggn256_batch262144.py"
python "$SOURCE/verify_transformer_noshared.py" | tee "$SMOKE_ROOT/unit.stdout"
grep -q TRANSFORMER_NOSHARED_VERIFY_OK "$SMOKE_ROOT/unit.stdout"

run_one() {
  local method=$1
  local trainer=$2
  local config=$3
  local seed=$4
  local workdir=$SMOKE_ROOT/${method}_seed${seed}_work
  mkdir -p "$workdir"
  ln -sfn "$SOURCE/configs" "$workdir/configs"
  (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES=0 python -u "$SOURCE/$trainer" \
      --config "$config" \
      --env_name halfcheetah \
      --seed "$seed" \
      --device 0 \
      --timesteps_per_proc 0
  ) > "$SMOKE_ROOT/${method}_seed${seed}.stdout" 2> "$SMOKE_ROOT/${method}_seed${seed}.stderr"
  if grep -Eiq "out of memory|(^|[^a-z])nan([^a-z]|$)|traceback|linalgerror" \
    "$SMOKE_ROOT/${method}_seed${seed}.stdout" "$SMOKE_ROOT/${method}_seed${seed}.stderr"; then
    echo "Error marker in $method seed $seed" >&2
    return 1
  fi
  find "$workdir/logs" -type f \( \
    \( -name progress.csv -size +0c \) -o \
    \( -name model.ckpt -size +0c \) \
  \) -print -quit | grep -q .
}

run_one ppo train_detach_ppo_public.py ppo_transformer_detach_fixedlr3e4_public.yaml 0
run_one kfac train_detach_ppo_public.py kfac_transformer_detach.yaml 0
run_one emp256 train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py rat_transformer_detjc_emp256_ggn256.yaml 0
run_one energyfree255p1 train_detach_energyfree255p1_criticggn256_batch262144.py rat_transformer_detjc_energyfree255p1_ggn256.yaml 0

run_one fullemp train_detach_jointcritic_actor_fvp_fisherclip.py rat_transformer_detjc_full_emp_full_ggn.yaml 0 &
pid0=$!
run_one fullemp train_detach_jointcritic_actor_fvp_fisherclip.py rat_transformer_detjc_full_emp_full_ggn.yaml 1 &
pid1=$!
wait "$pid0"
wait "$pid1"

nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu --format=csv > "$SMOKE_ROOT/gpu_end.csv"
echo FINISHED > "$SMOKE_ROOT/status"
