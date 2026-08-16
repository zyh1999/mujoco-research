#!/bin/bash
set -Eeuo pipefail

SOURCE=${SOURCE:-/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-transformer-noshared-20260813-v1}
PY=${PY:-/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python}
METHOD=${METHOD:?METHOD must be set}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be set}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
NUM_GPUS=${NUM_GPUS:-4}

case "$METHOD" in
  ppo)
    TRAINER=$SOURCE/train_detach_ppo_public.py
    CONFIG=ppo_transformer_detach_fixedlr3e4_public.yaml
    MAX_PARALLEL_PER_GPU=${MAX_PARALLEL_PER_GPU:-5}
    ;;
  kfac)
    TRAINER=$SOURCE/train_detach_ppo_public.py
    CONFIG=kfac_transformer_detach.yaml
    MAX_PARALLEL_PER_GPU=${MAX_PARALLEL_PER_GPU:-5}
    ;;
  emp256)
    TRAINER=$SOURCE/train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py
    CONFIG=rat_transformer_detjc_emp256_ggn256.yaml
    MAX_PARALLEL_PER_GPU=${MAX_PARALLEL_PER_GPU:-3}
    ;;
  energyfree255p1)
    TRAINER=$SOURCE/train_detach_energyfree255p1_criticggn256_batch262144.py
    CONFIG=rat_transformer_detjc_energyfree255p1_ggn256.yaml
    MAX_PARALLEL_PER_GPU=${MAX_PARALLEL_PER_GPU:-3}
    ;;
  fullemp)
    TRAINER=$SOURCE/train_detach_jointcritic_actor_fvp_fisherclip.py
    CONFIG=rat_transformer_detjc_full_emp_full_ggn.yaml
    MAX_PARALLEL_PER_GPU=${MAX_PARALLEL_PER_GPU:-2}
    ;;
  *)
    echo "unsupported METHOD=$METHOD" >&2
    exit 2
    ;;
esac

module load libs/gcc/glew/2.1.0 || true
export PATH=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SOURCE"
mkdir -p "$RUN_ROOT/workers" "$RUN_ROOT/task_locks"

ENVS=(ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d)
TASKS=$RUN_ROOT/tasks.tsv
{
  printf "shard\tenv\tseed\n"
  i=0
  for env in "${ENVS[@]}"; do
    for seed in 0 1 2 3 4; do
      printf "%s\t%s\t%s\n" "$((i % NUM_GPUS))" "$env" "$seed"
      i=$((i + 1))
    done
  done
} > "$TASKS"

{
  echo "method=$METHOD"
  echo "architecture=single_layer_mujoco_body_transformer"
  echo "network_mode=no_shared_independent_actor_critic"
  echo "launcher=csf3_gpuL4_multiworker"
  echo "gpu_workers=$NUM_GPUS"
  echo "max_parallel_per_gpu=$MAX_PARALLEL_PER_GPU"
  echo "environments=${ENVS[*]}"
  echo "seeds=0 1 2 3 4"
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
  echo "source=$SOURCE"
  echo "run_root=$RUN_ROOT"
  echo "trainer=$TRAINER"
  echo "config=$SOURCE/configs/$CONFIG"
  echo "trainer_sha256=$(sha256sum "$TRAINER" | awk '{print $1}')"
  echo "config_sha256=$(sha256sum "$SOURCE/configs/$CONFIG" | awk '{print $1}')"
  echo "transformer_sha256=$(sha256sum "$SOURCE/utils/mujoco_transformer.py" | awk '{print $1}')"
  echo "source_manifest_sha256=$(sha256sum "$SOURCE/source_manifest.sha256" | awk '{print $1}')"
  echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "host=$(hostname -f)"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$RUN_ROOT/run_info.txt"

echo RUNNING > "$RUN_ROOT/status"

run_seed() {
  local gpu=$1 env_name=$2 seed=$3
  local outdir=$RUN_ROOT/$METHOD/$env_name/seed${seed}
  local workdir=$outdir/work
  mkdir -p "$workdir"
  ln -sfn "$SOURCE/configs" "$workdir/configs"
  {
    echo "method=$METHOD"
    echo "environment=$env_name"
    echo "seed=$seed"
    echo "gpu_worker=$gpu"
    echo "source=$SOURCE"
    echo "run_root=$RUN_ROOT"
    echo "trainer=$TRAINER"
    echo "config=$CONFIG"
    echo "total_timesteps=$TOTAL_TIMESTEPS"
    echo "started_utc=$(date -u +%FT%TZ)"
  } > "$outdir/run_info.txt"
  echo RUNNING > "$outdir/status"
  (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$TRAINER" \
      --config "$CONFIG" \
      --env_name "$env_name" \
      --seed "$seed" \
      --device 0 \
      --timesteps_per_proc "$TOTAL_TIMESTEPS"
  ) > "$outdir/stdout.log" 2> "$outdir/stderr.log"
  local rc=$?
  echo "$rc" > "$outdir/rc"
  if [[ $rc -eq 0 ]] && grep -Eiq "out of memory|(^|[^a-z])nan([^a-z]|$)|traceback|linalgerror" \
      "$outdir/stdout.log" "$outdir/stderr.log"; then
    echo "error marker found in logs" >> "$outdir/stderr.log"
    rc=97
    echo "$rc" > "$outdir/rc"
  fi
  if [[ $rc -eq 0 ]] && ! find "$workdir/logs" -name progress.csv -type f -size +0c -print -quit | grep -q .; then
    echo "missing nonempty progress.csv" >> "$outdir/stderr.log"
    rc=98
    echo "$rc" > "$outdir/rc"
  fi
  if [[ $rc -eq 0 ]]; then
    echo FINISHED > "$outdir/status"
  else
    echo FAILED > "$outdir/status"
  fi
  return "$rc"
}

gpu_worker() {
  local gpu=$1
  local worker_dir=$RUN_ROOT/workers/gpu${gpu}
  mkdir -p "$worker_dir"
  echo RUNNING > "$worker_dir/status"
  echo "$$" > "$worker_dir/pid"
  {
    echo "gpu=$gpu"
    echo "method=$METHOD"
    echo "max_parallel_per_gpu=$MAX_PARALLEL_PER_GPU"
    echo "started_utc=$(date -u +%FT%TZ)"
  } > "$worker_dir/run_info.txt"

  local pids=()
  local labels=()
  local failed=0

  finish_first_child() {
    local pid=${pids[0]}
    local label=${labels[0]}
    local rc=0
    wait "$pid" || rc=$?
    printf "%s\tEND\t%s\trc=%s\n" "$(date -u +%FT%TZ)" "$label" "$rc" >> "$worker_dir/events.tsv"
    if [[ $rc -ne 0 ]]; then
      failed=1
      if [[ "${STOP_ON_FAILURE:-0}" == 1 ]]; then
        echo FAILED > "$worker_dir/status"
        exit "$rc"
      fi
    fi
    pids=("${pids[@]:1}")
    labels=("${labels[@]:1}")
  }

  while IFS=$'\t' read -r shard env_name seed; do
    [[ "$shard" != shard ]] || continue
    [[ "$shard" == "$gpu" ]] || continue
    local outdir=$RUN_ROOT/$METHOD/$env_name/seed${seed}
    local status_file=$outdir/status
    local task_id=${METHOD}_${env_name}_seed${seed}
    local lock_dir=$RUN_ROOT/task_locks/$task_id.lock
    if [[ -f "$status_file" ]]; then
      local status
      status=$(cat "$status_file")
      if [[ "$status" == FINISHED || "$status" == RUNNING ]]; then
        printf "%s\tSKIP_%s\t%s\n" "$(date -u +%FT%TZ)" "$status" "$task_id" >> "$worker_dir/events.tsv"
        continue
      fi
    fi
    if ! mkdir "$lock_dir" 2>/dev/null; then
      printf "%s\tSKIP_LOCKED\t%s\n" "$(date -u +%FT%TZ)" "$task_id" >> "$worker_dir/events.tsv"
      continue
    fi
    while [[ ${#pids[@]} -ge $MAX_PARALLEL_PER_GPU ]]; do
      finish_first_child
    done
    printf "%s\tSTART\t%s\n" "$(date -u +%FT%TZ)" "$task_id" >> "$worker_dir/events.tsv"
    (
      rc=0
      run_seed "$gpu" "$env_name" "$seed" || rc=$?
      rm -rf "$lock_dir"
      exit "$rc"
    ) &
    pids+=("$!")
    labels+=("$task_id")
    sleep 1
  done < "$TASKS"

  while [[ ${#pids[@]} -gt 0 ]]; do
    finish_first_child
  done

  if [[ $failed -eq 0 ]]; then
    echo FINISHED > "$worker_dir/status"
  else
    echo FAILED > "$worker_dir/status"
    exit 1
  fi
}

FAIL=0
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  gpu_worker "$gpu" > "$RUN_ROOT/workers/gpu${gpu}.stdout" 2> "$RUN_ROOT/workers/gpu${gpu}.stderr" &
  echo "$!" > "$RUN_ROOT/workers/gpu${gpu}.launcher_pid"
done

for pidfile in "$RUN_ROOT"/workers/gpu*.launcher_pid; do
  pid=$(cat "$pidfile")
  wait "$pid" || FAIL=1
done

if [[ $FAIL -eq 0 ]]; then
  echo FINISHED > "$RUN_ROOT/status"
else
  echo FAILED > "$RUN_ROOT/status"
  exit 1
fi
