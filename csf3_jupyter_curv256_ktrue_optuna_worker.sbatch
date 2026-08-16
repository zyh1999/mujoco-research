#!/usr/bin/env bash
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x-%j.err
#SBATCH --chdir=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original

set -euo pipefail

REPO=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
PY=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be exported}
ENVS=${ENVS:?ENVS must be colon-separated}
SHARD=${SHARD:?SHARD must name this worker}
TRAINER="$REPO/train_detach_jointcritic_curv256_ktrue_clip_momentum_optuna.py"
WORKER="$REPO/run_curv256_ktrue_clip_momentum_optuna_worker.py"
CONFIG=rat_mlp_detjc_curv256_criticggn_ktrue_clip_momentum_optuna_batch262144.yaml

mkdir -p "$RUN_ROOT"
printf 'STARTING\n' > "$RUN_ROOT/status_${SHARD}"
printf '%s\n' "$SLURM_JOB_ID" > "$RUN_ROOT/job_${SHARD}.txt"

export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load apps/binapps/jupyter-notebook/6.0.0
PORT=$((8800 + SLURM_JOB_ID % 1000))
NODE=$(hostname)
printf 'job=%s node=%s port=%s shard=%s envs=%s\n' \
  "$SLURM_JOB_ID" "$NODE" "$PORT" "$SHARD" "$ENVS" \
  | tee "$RUN_ROOT/allocation_${SHARD}.txt"

jupyter notebook --no-browser --ip=0.0.0.0 --port="$PORT" --notebook-dir="$REPO" \
  > "$RUN_ROOT/jupyter_${SHARD}.log" 2>&1 &
JUPYTER_PID=$!
trap 'kill "$JUPYTER_PID" 2>/dev/null || true' EXIT
printf '%s\n' "$JUPYTER_PID" > "$RUN_ROOT/jupyter_${SHARD}.pid"

IFS=: read -r -a ENV_ARRAY <<< "$ENVS"
printf 'RUNNING\n' > "$RUN_ROOT/status_${SHARD}"
"$PY" -u "$WORKER" \
  --gpu 0 \
  --envs "${ENV_ARRAY[@]}" \
  --repo "$REPO" \
  --root "$RUN_ROOT" \
  --trainer "$TRAINER" \
  --python "$PY" \
  --config "$CONFIG" \
  --new-trials 8 \
  --min-free-mb 7000
printf 'FINISHED\n' > "$RUN_ROOT/status_${SHARD}"
