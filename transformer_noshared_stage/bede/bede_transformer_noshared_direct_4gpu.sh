#!/usr/bin/env bash
# Run directly on a Bede GPU compute node that exposes at least four GPUs.
# This script intentionally does not invoke sbatch, srun, or salloc.
set -Eeuo pipefail

ROOT=/nobackup/projects/bdman37/yihe
SOURCE=${SOURCE:-$ROOT/src/ICML2026-RAT-transformer-noshared-20260813-v1}
CONDA_ROOT=$ROOT/ppc64le/miniconda
CONDA_ENV=$ROOT/ppc64le/envs/rat
GLFW_PREFIX=$ROOT/local/glfw-conda
NUM_GPUS=${NUM_GPUS:-4}
SEED_PARALLEL=${SEED_PARALLEL:-4}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
DIRECT_RUN_ROOT=${DIRECT_RUN_ROOT:-$ROOT/perf_runs/bede_transformer_noshared_direct4gpu_allmethods_all7_5seed_10m_$(date +%Y%m%d_%H%M%S)}
DEDUPE_ROOT=$ROOT/transformer_noshared_20260813/dedupe

METHODS=(ppo kfac emp256 energyfree255p1 fullemp)
ENVS=(ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d)

if [[ $NUM_GPUS -ne 4 ]]; then
  echo "This direct launcher is calibrated for exactly four GPUs; got NUM_GPUS=$NUM_GPUS" >&2
  exit 2
fi
if [[ $SEED_PARALLEL -ne 4 ]]; then
  echo "This direct launcher is calibrated for four concurrent seeds per GPU; got SEED_PARALLEL=$SEED_PARALLEL" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Run this from a Bede GPU compute node, not login1." >&2
  exit 3
fi
VISIBLE_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ $VISIBLE_GPUS -lt $NUM_GPUS ]]; then
  echo "Need $NUM_GPUS visible GPUs but found $VISIBLE_GPUS. Run this from a four-GPU Bede node." >&2
  exit 4
fi
if [[ ! -d $SOURCE ]] || [[ ! -x $CONDA_ROOT/bin/conda ]]; then
  echo "Missing source or Conda environment: SOURCE=$SOURCE CONDA_ROOT=$CONDA_ROOT" >&2
  exit 5
fi

module load gcc/12.2
module load cmake/3.18.4
module load cuda/12.4.1
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
PY=$(command -v python)

export LD_LIBRARY_PATH=$GLFW_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PIP_CACHE_DIR=$ROOT/ppc64le/pip-cache
export XDG_CACHE_HOME=$ROOT/ppc64le/xdg-cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$DIRECT_RUN_ROOT/workers" "$DIRECT_RUN_ROOT/task_locks" "$DEDUPE_ROOT"
nvidia-smi > "$DIRECT_RUN_ROOT/nvidia_smi_start.txt"

{
  echo "launcher=bede_direct_4gpu_no_slurm"
  echo "architecture=single_layer_mujoco_body_transformer"
  echo "network_mode=no_shared_independent_actor_critic"
  echo "methods=${METHODS[*]}"
  echo "environments=${ENVS[*]}"
  echo "seeds=0 1 2 3 4"
  echo "visible_gpus=$VISIBLE_GPUS"
  echo "used_gpus=$NUM_GPUS"
  echo "seed_parallel_per_gpu=$SEED_PARALLEL"
  echo "maximum_simultaneous_trainers=$((NUM_GPUS * SEED_PARALLEL))"
  echo "timesteps_per_seed=$TOTAL_TIMESTEPS"
  echo "rollout_samples=8192"
  echo "actor_epochs=4"
  echo "actor_minibatches=8"
  echo "critic_epochs=4"
  echo "critic_minibatches=8"
  echo "source=$SOURCE"
  echo "python=$PY"
  echo "launcher_sha256=$(sha256sum "$0" | awk '{print $1}')"
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname -f)"
} > "$DIRECT_RUN_ROOT/run_info.txt"
echo RUNNING > "$DIRECT_RUN_ROOT/status"

trainer_for_method() {
  local method=$1
  case "$method" in
    ppo)
      TRAINER=$SOURCE/train_detach_ppo_public.py
      CONFIG=ppo_transformer_detach_fixedlr3e4_public.yaml
      ;;
    kfac)
      TRAINER=$SOURCE/train_detach_ppo_public.py
      CONFIG=kfac_transformer_detach.yaml
      ;;
    emp256)
      TRAINER=$SOURCE/train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py
      CONFIG=rat_transformer_detjc_emp256_ggn256.yaml
      ;;
    energyfree255p1)
      TRAINER=$SOURCE/train_detach_energyfree255p1_criticggn256_batch262144.py
      CONFIG=rat_transformer_detjc_energyfree255p1_ggn256.yaml
      ;;
    fullemp)
      TRAINER=$SOURCE/train_detach_jointcritic_actor_fvp_fisherclip.py
      CONFIG=rat_transformer_detjc_full_emp_full_ggn.yaml
      ;;
    *)
      echo "unsupported method=$method" >&2
      return 2
      ;;
  esac
}

run_seed() {
  local gpu=$1 method=$2 env_name=$3 seed=$4 outdir=$5
  trainer_for_method "$method"
  local seed_dir=$outdir/seed${seed}
  local workdir=$seed_dir/work
  mkdir -p "$workdir"
  ln -sfn "$SOURCE/configs" "$workdir/configs"
  {
    echo "method=$method"
    echo "environment=$env_name"
    echo "seed=$seed"
    echo "gpu=$gpu"
    echo "trainer=$TRAINER"
    echo "config=$CONFIG"
    echo "timesteps=$TOTAL_TIMESTEPS"
    echo "started=$(date --iso-8601=seconds)"
  } > "$seed_dir/run_info.txt"
  echo RUNNING > "$seed_dir/status"
  local rc=0
  if (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID=0 "$PY" -u "$TRAINER" \
      --config "$CONFIG" --env_name "$env_name" --seed "$seed" --device 0 \
      --timesteps_per_proc "$TOTAL_TIMESTEPS"
  ) > "$seed_dir/stdout.log" 2> "$seed_dir/stderr.log"; then
    rc=0
  else
    rc=$?
  fi
  if [[ $rc -eq 0 ]] && grep -Eiq "out of memory|(^|[^a-z])nan([^a-z]|$)|traceback|linalgerror" \
      "$seed_dir/stdout.log" "$seed_dir/stderr.log"; then
    rc=97
    echo "error marker found in logs" >> "$seed_dir/stderr.log"
  fi
  if [[ $rc -eq 0 ]] && ! find "$workdir/logs" -name progress.csv -type f -size +0c -print -quit | grep -q .; then
    rc=98
    echo "missing nonempty progress.csv" >> "$seed_dir/stderr.log"
  fi
  echo "$rc" > "$seed_dir/rc"
  if [[ $rc -eq 0 ]]; then
    echo FINISHED > "$seed_dir/status"
  else
    echo FAILED > "$seed_dir/status"
  fi
  return "$rc"
}

run_method_env() {
  local gpu=$1 method=$2 env_name=$3
  local outdir=$DIRECT_RUN_ROOT/$method/$env_name
  local dedupe_key=${method}_${env_name}
  local lockfile=$DEDUPE_ROOT/${dedupe_key}.lock
  local donefile=$DEDUPE_ROOT/${dedupe_key}.done
  mkdir -p "$outdir"

  if [[ -f $donefile ]]; then
    echo SKIPPED_DONE > "$outdir/status"
    return 0
  fi

  (
    exec 9>"$lockfile"
    if ! flock -n 9; then
      echo SKIPPED_LOCKED > "$outdir/status"
      exit 0
    fi
    if [[ -f $donefile ]]; then
      echo SKIPPED_DONE > "$outdir/status"
      exit 0
    fi

    {
      echo "method=$method"
      echo "environment=$env_name"
      echo "gpu=$gpu"
      echo "seed_parallel=$SEED_PARALLEL"
      echo "dedupe_lock=$lockfile"
      echo "started=$(date --iso-8601=seconds)"
    } > "$outdir/run_info.txt"
    printf "seed\tstatus\trc\n" > "$outdir/seed_status.tsv"
    echo RUNNING > "$outdir/status"

    local pids=() seeds=() failed=0
    finish_one() {
      local pid=${pids[0]} seed=${seeds[0]} rc=0
      wait "$pid" || rc=$?
      if [[ $rc -eq 0 ]]; then
        printf "%s\tFINISHED\t0\n" "$seed" >> "$outdir/seed_status.tsv"
      else
        printf "%s\tFAILED\t%s\n" "$seed" "$rc" >> "$outdir/seed_status.tsv"
        failed=1
      fi
      pids=("${pids[@]:1}")
      seeds=("${seeds[@]:1}")
    }

    for seed in 0 1 2 3 4; do
      while [[ ${#pids[@]} -ge $SEED_PARALLEL ]]; do
        finish_one
      done
      run_seed "$gpu" "$method" "$env_name" "$seed" "$outdir" &
      pids+=("$!")
      seeds+=("$seed")
      sleep 1
    done
    while [[ ${#pids[@]} -gt 0 ]]; do
      finish_one
    done

    if [[ $failed -eq 0 ]]; then
      printf "method=%s\nenvironment=%s\nrun_root=%s\nfinished=%s\n" \
        "$method" "$env_name" "$DIRECT_RUN_ROOT" "$(date --iso-8601=seconds)" > "$donefile"
      echo FINISHED > "$outdir/status"
    else
      echo FAILED > "$outdir/status"
      exit 1
    fi
  )
}

TASKS=$DIRECT_RUN_ROOT/tasks.tsv
{
  printf "gpu\tmethod\tenv\n"
  i=0
  for env_name in "${ENVS[@]}"; do
    for method in "${METHODS[@]}"; do
      printf "%s\t%s\t%s\n" "$((i % NUM_GPUS))" "$method" "$env_name"
      i=$((i + 1))
    done
  done
} > "$TASKS"

gpu_worker() {
  local gpu=$1 worker_dir=$DIRECT_RUN_ROOT/workers/gpu${gpu}
  mkdir -p "$worker_dir"
  echo RUNNING > "$worker_dir/status"
  echo "$$" > "$worker_dir/pid"
  local failed=0
  while IFS=$'\t' read -r shard method env_name; do
    [[ $shard != gpu ]] || continue
    [[ $shard == "$gpu" ]] || continue
    local rc=0
    run_method_env "$gpu" "$method" "$env_name" || rc=$?
    printf "%s\t%s\t%s\trc=%s\n" "$(date --iso-8601=seconds)" "$method" "$env_name" "$rc" \
      >> "$worker_dir/events.tsv"
    [[ $rc -eq 0 ]] || failed=1
  done < "$TASKS"
  if [[ $failed -eq 0 ]]; then
    echo FINISHED > "$worker_dir/status"
  else
    echo FAILED > "$worker_dir/status"
    return 1
  fi
}

for gpu in 0 1 2 3; do
  gpu_worker "$gpu" > "$DIRECT_RUN_ROOT/workers/gpu${gpu}.stdout" 2> "$DIRECT_RUN_ROOT/workers/gpu${gpu}.stderr" &
  echo "$!" > "$DIRECT_RUN_ROOT/workers/gpu${gpu}.launcher_pid"
done

failed=0
for pidfile in "$DIRECT_RUN_ROOT"/workers/gpu*.launcher_pid; do
  pid=$(cat "$pidfile")
  wait "$pid" || failed=1
done
nvidia-smi > "$DIRECT_RUN_ROOT/nvidia_smi_end.txt" || true
if [[ $failed -eq 0 ]]; then
  echo FINISHED > "$DIRECT_RUN_ROOT/status"
else
  echo FAILED > "$DIRECT_RUN_ROOT/status"
  exit 1
fi
