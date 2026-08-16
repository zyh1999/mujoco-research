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
SOURCE_RUN_ROOT="$REPO/perf_runs/csf3_detach_mujoco_batch262144_curv256ggn_ktrue_l2clip05_fvpclip05_all7_5seed_500u_gpuL2_20260805"
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be exported}
PACK_LIST=${PACK_LIST:?PACK_LIST must be colon-separated task indices}
SHARD=${SHARD:?SHARD must name this allocation}

mkdir -p "$RUN_ROOT" "$REPO/slurm_logs"
printf 'STARTING\n' > "$RUN_ROOT/status_${SHARD}"
printf '%s\n' "$SLURM_JOB_ID" > "$RUN_ROOT/job_${SHARD}.txt"
printf '%s\n' "$PACK_LIST" > "$RUN_ROOT/packs_${SHARD}.txt"

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
printf 'job=%s node=%s port=%s shard=%s packs=%s\n' \
  "$SLURM_JOB_ID" "$NODE" "$PORT" "$SHARD" "$PACK_LIST" \
  | tee "$RUN_ROOT/allocation_${SHARD}.txt"
nvidia-smi > "$RUN_ROOT/nvidia_smi_start_${SHARD}.txt"

jupyter notebook \
  --no-browser \
  --ip=0.0.0.0 \
  --port="$PORT" \
  --notebook-dir="$REPO" \
  > "$RUN_ROOT/jupyter_${SHARD}.log" 2>&1 &
JUPYTER_PID=$!
trap 'kill "$JUPYTER_PID" 2>/dev/null || true' EXIT
printf '%s\n' "$JUPYTER_PID" > "$RUN_ROOT/jupyter_${SHARD}.pid"

printf 'RUNNING\n' > "$RUN_ROOT/status_${SHARD}"
IFS=: read -r -a PACKS <<< "$PACK_LIST"
for TASK_INDEX in "${PACKS[@]}"; do
  printf 'task=%s start=%s\n' "$TASK_INDEX" "$(date -Is)" \
    | tee -a "$RUN_ROOT/timeline_${SHARD}.txt"
  "$PY" -u "$SOURCE_RUN_ROOT/run_batch262144_clipped_pack5_worker.py" \
    --run-root "$RUN_ROOT" \
    --task-index "$TASK_INDEX" \
    --python "$PY" \
    --repo "$REPO"
  printf 'task=%s finish=%s\n' "$TASK_INDEX" "$(date -Is)" \
    | tee -a "$RUN_ROOT/timeline_${SHARD}.txt"
done

nvidia-smi > "$RUN_ROOT/nvidia_smi_finish_${SHARD}.txt"
printf 'FINISHED\n' > "$RUN_ROOT/status_${SHARD}"
