#!/bin/bash
#SBATCH --job-name=mj-kopt-4v100
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=480G
#SBATCH --gres=full:1,gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%j.err

set -Eeuo pipefail

ROOT=/nobackup/projects/bdman37/yihe
REPO="$ROOT/src/ICML2026-RAT"
RUN_ROOT="$ROOT/perf_runs/bede_detjc_batch262144_curv256ggn_kopt_clip_momentum_optuna_2seed_20260805"
PY="$ROOT/ppc64le/envs/rat/bin/python"
TRAINER="$REPO/train_detach_jointcritic_curv256_kopt_bede.py"
WORKER="$REPO/run_curv256_ktrue_clip_momentum_optuna_worker.py"
CONFIG=rat_mlp_detjc_curv256_criticggn_kopt_batch262144_bede.yaml

mkdir -p "$RUN_ROOT" "$ROOT/logs" "$ROOT/tmp/$SLURM_JOB_ID"
printf '%s\n' "$SLURM_JOB_ID" >> "$RUN_ROOT/job_ids.txt"
printf 'STARTING\n' > "$RUN_ROOT/status"

module load gcc/12.2
module load cmake/3.18.4
module load cuda/12.4.1
source "$ROOT/ppc64le/miniconda/etc/profile.d/conda.sh"
conda activate "$ROOT/ppc64le/envs/rat"

export PYTHONPATH="$ROOT/ppc64le/python_packages/optuna_4_5${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$ROOT/tmp/$SLURM_JOB_ID"
export PIP_CACHE_DIR="$ROOT/ppc64le/pip-cache"
export XDG_CACHE_HOME="$ROOT/ppc64le/xdg-cache"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PY" - <<'PY'
import optuna
import torch
import yaml
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
print("optuna", optuna.__version__)
print("torch", torch.__version__)
for index in range(4):
    print("gpu", index, torch.cuda.get_device_name(index))
PY

launch_worker() {
  local gpu=$1
  local shard=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$WORKER" \
    --gpu "$gpu" \
    --envs "$@" \
    --repo "$REPO" \
    --root "$RUN_ROOT" \
    --trainer "$TRAINER" \
    --python "$PY" \
    --config "$CONFIG" \
    --new-trials 8 \
    --min-free-mb 12000 \
    --storage-kind journal \
    > "$RUN_ROOT/worker_${shard}.stdout" \
    2> "$RUN_ROOT/worker_${shard}.stderr" &
  printf '%s\n' "$!" > "$RUN_ROOT/worker_${shard}.pid"
}

printf 'RUNNING\n' > "$RUN_ROOT/status"
launch_worker 0 ant_humanoid ant humanoid
launch_worker 1 halfcheetah_walker halfcheetah walker2d
launch_worker 2 hopper_hstandup hopper humanoidstandup
launch_worker 3 swimmer swimmer

rc=0
for pid_file in "$RUN_ROOT"/worker_*.pid; do
  pid=$(cat "$pid_file")
  if ! wait "$pid"; then
    rc=1
  fi
done

if (( rc == 0 )); then
  printf 'FINISHED\n' > "$RUN_ROOT/status"
else
  printf 'FAILED_OR_INTERRUPTED\n' > "$RUN_ROOT/status"
fi
exit "$rc"
