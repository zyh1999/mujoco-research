#!/bin/bash
set -Eeuo pipefail

ROOT=/nobackup/projects/bdman37/yihe
SOURCE=${SOURCE:-$ROOT/src/ICML2026-RAT-transformer-noshared-20260813-v1}
CONDA_ROOT=$ROOT/ppc64le/miniconda
CONDA_ENV=$ROOT/ppc64le/envs/rat
GLFW_PREFIX=$ROOT/local/glfw-conda
METHOD=${METHOD:?METHOD must be set}
ENV_NAME=${ENV_NAME:?ENV_NAME must be set}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be set}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
SEED_LIST=${SEED_LIST:-"0 1 2 3 4"}

case "$METHOD" in
  ppo)
    TRAINER=$SOURCE/train_detach_ppo_public.py
    CONFIG=ppo_transformer_detach_fixedlr3e4_public.yaml
    MAX_PARALLEL=${MAX_PARALLEL:-5}
    ;;
  kfac)
    TRAINER=$SOURCE/train_detach_ppo_public.py
    CONFIG=kfac_transformer_detach.yaml
    MAX_PARALLEL=${MAX_PARALLEL:-5}
    ;;
  emp256)
    TRAINER=$SOURCE/train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py
    CONFIG=rat_transformer_detjc_emp256_ggn256.yaml
    MAX_PARALLEL=${MAX_PARALLEL:-3}
    ;;
  energyfree255p1)
    TRAINER=$SOURCE/train_detach_energyfree255p1_criticggn256_batch262144.py
    CONFIG=rat_transformer_detjc_energyfree255p1_ggn256.yaml
    MAX_PARALLEL=${MAX_PARALLEL:-3}
    ;;
  fullemp)
    TRAINER=$SOURCE/train_detach_jointcritic_actor_fvp_fisherclip.py
    CONFIG=rat_transformer_detjc_full_emp_full_ggn.yaml
    MAX_PARALLEL=${MAX_PARALLEL:-2}
    ;;
  *)
    echo "Unsupported METHOD=$METHOD" >&2
    exit 2
    ;;
esac

module load gcc/12.2
module load cmake/3.18.4
module load cuda/12.4.1
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR=$ROOT/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}
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
mkdir -p "$TMPDIR" "$ROOT/logs" "$RUN_ROOT"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "allocated V100 is not visible"
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0), flush=True)
PY

OUTDIR=$RUN_ROOT/$ENV_NAME
mkdir -p "$OUTDIR"

{
  echo "method=$METHOD"
  echo "architecture=single_layer_mujoco_body_transformer"
  echo "network_mode=no_shared_independent_actor_critic"
  echo "environment=$ENV_NAME"
  echo "seeds=$SEED_LIST"
  echo "seed_execution=bounded_parallel_on_one_gpu"
  echo "max_parallel_children=$MAX_PARALLEL"
  echo "timesteps_per_seed=$TOTAL_TIMESTEPS"
  echo "rollout_samples=8192"
  echo "actor_epochs=4"
  echo "actor_minibatches=8"
  echo "critic_epochs=4"
  echo "critic_minibatches=8"
  echo "transformer_layers=1"
  echo "transformer_width=64"
  echo "transformer_heads=4"
  echo "transformer_ff_multiplier=2"
  echo "trainer=$TRAINER"
  echo "config=$SOURCE/configs/$CONFIG"
  echo "trainer_sha256=$(sha256sum "$TRAINER" | awk '{print $1}')"
  echo "config_sha256=$(sha256sum "$SOURCE/configs/$CONFIG" | awk '{print $1}')"
  echo "transformer_sha256=$(sha256sum "$SOURCE/utils/mujoco_transformer.py" | awk '{print $1}')"
  echo "source_manifest_sha256=$(sha256sum "$SOURCE/source_manifest.sha256" | awk '{print $1}')"
  echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-none}"
  echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
  echo "host=$(hostname -f)"
  echo "started=$(date --iso-8601=seconds)"
} > "$OUTDIR/run_info.txt"

printf "seed\tstatus\trc\tworkdir\n" > "$OUTDIR/seed_status.tsv"
echo RUNNING > "$OUTDIR/status"

pids=()
seeds=()

cleanup_children() {
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup_children TERM INT

finish_first_child() {
  local pid=${pids[0]}
  local seed=${seeds[0]}
  local rc=0
  wait "$pid" || rc=$?
  if [[ $rc -eq 0 ]] && grep -Eiq "out of memory|(^|[^a-z])nan([^a-z]|$)|traceback|linalgerror" \
    "$OUTDIR/seed${seed}.stdout" "$OUTDIR/seed${seed}.stderr"; then
    rc=97
  fi
  if [[ $rc -eq 0 ]] && ! find "$OUTDIR/seed${seed}_work/logs" \
    -name progress.csv -type f -size +0c -print -quit | grep -q .; then
    rc=98
  fi
  echo "$rc" > "$OUTDIR/seed${seed}.rc"
  if [[ $rc -eq 0 ]]; then
    echo FINISHED > "$OUTDIR/seed${seed}.status"
    printf "%s\tFINISHED\t0\t%s\n" "$seed" "$OUTDIR/seed${seed}_work" >> "$OUTDIR/seed_status.tsv"
  else
    echo FAILED > "$OUTDIR/seed${seed}.status"
    printf "%s\tFAILED\t%s\t%s\n" "$seed" "$rc" "$OUTDIR/seed${seed}_work" >> "$OUTDIR/seed_status.tsv"
    failed=1
  fi
  pids=("${pids[@]:1}")
  seeds=("${seeds[@]:1}")
}

failed=0
for seed in $SEED_LIST; do
  while [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; do
    finish_first_child
  done

  workdir=$OUTDIR/seed${seed}_work
  mkdir -p "$workdir"
  ln -sfn "$SOURCE/configs" "$workdir/configs"
  echo RUNNING > "$OUTDIR/seed${seed}.status"
  (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES=0 python -u "$TRAINER" \
      --config "$CONFIG" \
      --env_name "$ENV_NAME" \
      --seed "$seed" \
      --device 0 \
      --timesteps_per_proc "$TOTAL_TIMESTEPS"
  ) > "$OUTDIR/seed${seed}.stdout" 2> "$OUTDIR/seed${seed}.stderr" &
  pids+=("$!")
  seeds+=("$seed")
  echo "$!" > "$OUTDIR/seed${seed}.pid"
  sleep 1
done

while [[ ${#pids[@]} -gt 0 ]]; do
  finish_first_child
done

if [[ $failed -eq 0 ]]; then
  echo FINISHED > "$OUTDIR/status"
else
  echo FAILED > "$OUTDIR/status"
  exit 1
fi
