#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yihe/ICML2026-RAT-original-4090
PY=/home/yihe/.venv/bin/python
TRAINER=train_detach_jointcritic.py
CONFIG=rat_mlp_empfullrhs_adam_kfalse.yaml
CONFIG_PATH=configs/$CONFIG
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be set}
TIMESTEPS=10000000
ENVS=(ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d)
SEEDS=(0 1 2 3 4)

cd "$ROOT"
mkdir -p "$RUN_ROOT"

export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH=/home/yihe/.mujoco/mujoco210
export LD_LIBRARY_PATH=/home/yihe/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
  echo "family=no_shared_emp64_emp256_fullrhs_adamcritic_kfalse"
  echo "host=$(hostname)"
  echo "trainer=$TRAINER"
  echo "trainer_sha256=$(sha256sum "$TRAINER" | awk '{print $1}')"
  echo "config=$CONFIG_PATH"
  echo "config_sha256=$(sha256sum "$CONFIG_PATH" | awk '{print $1}')"
  echo "python=$PY"
  echo "python_version=$($PY --version 2>&1)"
  echo "actor_network=separate"
  echo "critic_network=separate"
  echo "actor_rhs=full_minibatch_policy_gradient"
  echo "actor_subsample_full_batch_gradient=true"
  echo "actor_subsample_full_batch_gradient_raw_fisher=false"
  echo "critic_update=standard"
  echo "critic_optimizer=adam"
  echo "critic_curvature_subsample=0"
  echo "fisher_kernel=exact"
  echo "fisher_kernel_normalization=none"
  echo "kaczmarz=false"
  echo "damping=0.1"
  echo "lr_pi=0.05"
  echo "lr_v=0.001"
  echo "pi_epochs=4"
  echo "pi_minibatches=8"
  echo "v_epochs=4"
  echo "v_minibatches=8"
  echo "actor_clip=parameter_l2_0.5"
  echo "timesteps=$TIMESTEPS"
  echo "seeds=0,1,2,3,4"
  echo "envs=ant,halfcheetah,hopper,humanoid,humanoidstandup,swimmer,walker2d"
  echo "gpu0=emp64"
  echo "gpu1=emp256"
  echo "created_utc=$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader
} > "$RUN_ROOT/run_info.txt"

printf "variant\tenv\tseed\tgpu\tcurvature_rows\tstatus\trc\tstart_utc\tend_utc\n" > "$RUN_ROOT/manifest.tsv"

common_args=(
  --config "$CONFIG"
  --device 0
  --pi_epochs 4
  --lr_pi 0.05
  --cg_damping 0.1
  --fisher_kernel exact
  --fisher_kernel_normalization none
  --critic_curvature_subsample 0
  --critic_update standard
  --critic_optimizer adam
  --actor_subsample_full_batch_gradient
  --actor_subsample_free_rho none
)

run_preflight_one() {
  local gpu=$1
  local curvature=$2
  local env=$3
  local seed=$4
  local tag="emp${curvature}_${env}"
  mkdir -p "$RUN_ROOT/preflight"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$TRAINER" "${common_args[@]}" \
    --actor_curvature_subsample "$curvature" \
    --env_name "$env" --seed "$seed" --timesteps_per_proc 16384 \
    > "$RUN_ROOT/preflight/${tag}.stdout" \
    2> "$RUN_ROOT/preflight/${tag}.stderr"
}

preflight_rc64=0
preflight_rc256=0
run_preflight_one 0 64 swimmer 9064 || preflight_rc64=$?
run_preflight_one 1 256 halfcheetah 9256 || preflight_rc256=$?
printf "emp64_gpu0_rc=%s\nemp256_gpu1_rc=%s\n" "$preflight_rc64" "$preflight_rc256" > "$RUN_ROOT/preflight/status.txt"
if [[ $preflight_rc64 -ne 0 || $preflight_rc256 -ne 0 ]]; then
  echo PREFLIGHT_FAILED > "$RUN_ROOT/status"
  exit 2
fi

run_env() {
  local gpu=$1
  local curvature=$2
  local env=$3
  local variant="emp${curvature}_kfalse"
  local outdir="$RUN_ROOT/$variant/$env"
  mkdir -p "$outdir"
  echo RUNNING > "$outdir/status"

  local pids=()
  local started=()
  local seed
  for seed in "${SEEDS[@]}"; do
    started+=("$(date -u +%FT%TZ)")
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$TRAINER" "${common_args[@]}" \
      --actor_curvature_subsample "$curvature" \
      --env_name "$env" --seed "$seed" --timesteps_per_proc "$TIMESTEPS" \
      > "$outdir/seed${seed}.stdout" \
      2> "$outdir/seed${seed}.stderr" &
    pids+=("$!")
    echo "$!" > "$outdir/seed${seed}.pid"
    printf "%q " "$PY" -u "$TRAINER" "${common_args[@]}" \
      --actor_curvature_subsample "$curvature" \
      --env_name "$env" --seed "$seed" --timesteps_per_proc "$TIMESTEPS" \
      > "$outdir/seed${seed}.command.txt"
    printf "\n" >> "$outdir/seed${seed}.command.txt"
    sleep 1
  done

  local failed=0
  local i rc end
  for i in "${!pids[@]}"; do
    rc=0
    wait "${pids[$i]}" || rc=$?
    seed=${SEEDS[$i]}
    end=$(date -u +%FT%TZ)
    echo "$rc" > "$outdir/seed${seed}.rc"
    if [[ $rc -eq 0 ]]; then
      printf "%s\t%s\t%s\t%s\t%s\tFINISHED\t0\t%s\t%s\n" \
        "$variant" "$env" "$seed" "$gpu" "$curvature" "${started[$i]}" "$end" >> "$RUN_ROOT/manifest.tsv"
    else
      failed=1
      printf "%s\t%s\t%s\t%s\t%s\tFAILED\t%s\t%s\t%s\n" \
        "$variant" "$env" "$seed" "$gpu" "$curvature" "$rc" "${started[$i]}" "$end" >> "$RUN_ROOT/manifest.tsv"
    fi
  done

  if [[ $failed -eq 0 ]]; then
    echo FINISHED > "$outdir/status"
  else
    echo FAILED > "$outdir/status"
  fi
  return 0
}

run_variant() {
  local gpu=$1
  local curvature=$2
  local env
  for env in "${ENVS[@]}"; do
    run_env "$gpu" "$curvature" "$env"
  done
}

echo RUNNING > "$RUN_ROOT/status"
run_variant 0 64 > "$RUN_ROOT/emp64.worker.stdout" 2> "$RUN_ROOT/emp64.worker.stderr" &
worker64=$!
echo "$worker64" > "$RUN_ROOT/emp64.worker.pid"
run_variant 1 256 > "$RUN_ROOT/emp256.worker.stdout" 2> "$RUN_ROOT/emp256.worker.stderr" &
worker256=$!
echo "$worker256" > "$RUN_ROOT/emp256.worker.pid"

wait "$worker64"
wait "$worker256"

if find "$RUN_ROOT" -mindepth 3 -maxdepth 3 -name status -exec grep -l '^FAILED$' {} + | grep -q .; then
  echo PARTIAL > "$RUN_ROOT/status"
else
  echo FINISHED > "$RUN_ROOT/status"
fi
