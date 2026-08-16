#!/bin/bash
set -euo pipefail

ROOT=/home/yihe/ICML2026-RAT-original-4090
PY=/home/yihe/.venv/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be set}
TRAINER=train_detach_jointcritic_momentum09.py
CONFIG=rat_mlp_detjc_full_ef_full_ggn_momentum09_1024x8_e4.yaml
ENVS_GPU0=(ant hopper swimmer walker2d)
ENVS_GPU1=(halfcheetah humanoid humanoidstandup)

cd "$ROOT"
[[ "$(cat "$RUN_ROOT/preflight/status" 2>/dev/null)" == FINISHED ]]

export PATH=/home/yihe/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cat > "$RUN_ROOT/run_info.txt" <<EOF
family=no_shared_full_ef_full_ggn_momentum09
mode=no_shared
trainer=$TRAINER
config=$CONFIG
actor_curvature=full_empirical_fisher
actor_curvature_rows=1024
critic_update=independent_full_ggn
critic_curvature_rows=1024
rollout_samples=8192
rollout_layout=32_envs_x_256_steps
minibatches=8
minibatch_size=1024
pi_epochs=4
v_epochs=4
actor_sgd_momentum=0.9
critic_sgd_momentum=0.9
lr_pi=0.05
lr_v=0.1
damping=0.03
normalization=none
kaczmarz=false
fisher_kernel=exact
actor_subsample_full_batch_gradient=false
envs=ant,halfcheetah,hopper,humanoid,humanoidstandup,swimmer,walker2d
seeds=0,1,2,3,4
timesteps=10000000
host=$(hostname)
created_utc=$(date -u +%FT%TZ)
EOF

printf "env\tseeds\ttimesteps\trollout\tminibatch\tpi_epochs\tactor_rows\tcritic_rows\tactor_momentum\tcritic_momentum\tdamping\tnormalization\tkaczmarz\tgpu\n" > "$RUN_ROOT/manifest.tsv"
for env in "${ENVS_GPU0[@]}"; do
  printf "%s\t0,1,2,3,4\t10000000\t8192\t1024\t4\t1024\t1024\t0.9\t0.9\t0.03\tnone\tfalse\t0\n" "$env" >> "$RUN_ROOT/manifest.tsv"
done
for env in "${ENVS_GPU1[@]}"; do
  printf "%s\t0,1,2,3,4\t10000000\t8192\t1024\t4\t1024\t1024\t0.9\t0.9\t0.03\tnone\tfalse\t1\n" "$env" >> "$RUN_ROOT/manifest.tsv"
done

command_for() {
  local env=$1 seed=$2
  printf '%q ' "$PY" -u "$TRAINER" \
    --config "$CONFIG" --env_name "$env" --seed "$seed" --device 0 \
    --timesteps_per_proc 10000000 --pi_epochs 4 --lr_pi 0.05 \
    --cg_damping 0.03 --fisher_kernel exact \
    --fisher_kernel_normalization none \
    --actor_curvature_subsample 0 --critic_curvature_subsample 0 \
    --critic_update gn --critic_optimizer sgd \
    --no-actor_subsample_full_batch_gradient
}

run_one() {
  local gpu=$1 env=$2 seed=$3 outdir=$4
  local command
  command=$(command_for "$env" "$seed")
  printf '%s\n' "$command" > "$outdir/seed${seed}.command.txt"
  CUDA_VISIBLE_DEVICES=$gpu bash -lc "$command" > "$outdir/seed${seed}.stdout" 2> "$outdir/seed${seed}.stderr"
}

run_env() {
  local gpu=$1 env=$2 outdir="$RUN_ROOT/$2"
  mkdir -p "$outdir"
  echo RUNNING > "$outdir/status"
  local pids=() seeds=()
  for seed in 0 1 2 3 4; do
    if [[ -f "$outdir/seed${seed}.rc" ]] && [[ "$(cat "$outdir/seed${seed}.rc")" == 0 ]]; then
      continue
    fi
    run_one "$gpu" "$env" "$seed" "$outdir" &
    pids+=("$!")
    seeds+=("$seed")
    printf '%s\n' "$!" > "$outdir/seed${seed}.pid"
    sleep 1
  done
  local failed=0
  for i in "${!pids[@]}"; do
    local rc=0
    wait "${pids[$i]}" || rc=$?
    printf '%s\n' "$rc" > "$outdir/seed${seeds[$i]}.rc"
    [[ $rc -eq 0 ]] || failed=1
  done
  if [[ $failed -eq 0 ]]; then echo FINISHED > "$outdir/status"; else echo FAILED > "$outdir/status"; return 1; fi
}

run_queue() {
  local gpu=$1
  shift
  for env in "$@"; do run_env "$gpu" "$env"; done
}

echo RUNNING > "$RUN_ROOT/status"
run_queue 0 "${ENVS_GPU0[@]}" & worker0=$!
run_queue 1 "${ENVS_GPU1[@]}" & worker1=$!
rc=0
wait "$worker0" || rc=1
wait "$worker1" || rc=1
if [[ $rc -eq 0 ]]; then echo FINISHED > "$RUN_ROOT/status"; else echo FAILED > "$RUN_ROOT/status"; fi
exit "$rc"

