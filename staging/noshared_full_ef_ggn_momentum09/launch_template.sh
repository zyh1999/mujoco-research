#!/bin/bash
set -euo pipefail

ROOT=/home/yihe/ICML2026-RAT-original-4090
PY=/home/yihe/.venv/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be set}
MODE=${MODE:?MODE must be shared or no_shared}
DAMPING=${DAMPING:-0.03}
ENVS_GPU0=(ant hopper swimmer walker2d)
ENVS_GPU1=(halfcheetah humanoid humanoidstandup)

if [[ "$MODE" != shared && "$MODE" != no_shared ]]; then
  echo "Unsupported MODE=$MODE" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/preflight"
cd "$ROOT"

export PATH=/home/yihe/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ "$MODE" == shared ]]; then
  TRAINER=train_shared_jointkernel_energyfree.py
  CONFIG=rat_mlp_shared.yaml
  CRITIC_DESCRIPTION="joint shared per-sample GGN"
else
  TRAINER=train_detach_jointcritic.py
  CONFIG=rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml
  CRITIC_DESCRIPTION="independent per-sample GGN"
fi

cat > "$RUN_ROOT/run_info.txt" <<EOF
family=${MODE}_groupedrho4_ggn_normnone_damp${DAMPING}_kfalse
mode=${MODE}
trainer=${TRAINER}
config=${CONFIG}
actor_dof=256
actor_groups=256
actor_group_size=4
actor_group_constraint=alpha_i=rho_g*b_i
actor_group_partition=random_per_update
actor_group_kernel_denominator=256
actor_group_damping_metric=sum_group_b_squared
actor_ratio_location=trial_basis_inside_inverse
critic_update=${CRITIC_DESCRIPTION}
critic_curvature_subsample=256
envs=ant,halfcheetah,hopper,humanoid,humanoidstandup,swimmer,walker2d
seeds=0,1,2,3,4
timesteps=10000000
epochs=4
lr_pi=0.05
lr_v=0.1
damping=${DAMPING}
normalization=none
kaczmarz=false
job_id=${SLURM_JOB_ID:-manual}
host=$(hostname)
created_utc=$(date -u +%FT%TZ)
EOF

printf "mode\tenv\tseeds\ttimesteps\tepochs\tlr_pi\tlr_v\tdamping\tactor_dof\tactor_groups\tgroup_size\tcritic_rows\tnormalization\tkaczmarz\tgpu\n" > "$RUN_ROOT/manifest.tsv"
for env in "${ENVS_GPU0[@]}"; do
  printf "%s\t%s\t0,1,2,3,4\t10000000\t4\t0.05\t0.1\t%s\t256\t256\t4\t256\tnone\tfalse\t0\n" "$MODE" "$env" "$DAMPING" >> "$RUN_ROOT/manifest.tsv"
done
for env in "${ENVS_GPU1[@]}"; do
  printf "%s\t%s\t0,1,2,3,4\t10000000\t4\t0.05\t0.1\t%s\t256\t256\t4\t256\tnone\tfalse\t1\n" "$MODE" "$env" "$DAMPING" >> "$RUN_ROOT/manifest.tsv"
done

command_for() {
  local env=$1
  local seed=$2
  local timesteps=$3
  if [[ "$MODE" == shared ]]; then
    printf '%q ' "$PY" -u "$TRAINER" \
      --config "$CONFIG" --env_name "$env" --seed "$seed" --device 0 \
      --timesteps_per_proc "$timesteps" --epochs 4 --lr 0.05 \
      --cg_damping "$DAMPING" --fisher_kernel exact \
      --fisher_kernel_normalization none \
      --actor_curvature_subsample 256 --critic_curvature_subsample 256 \
      --actor_group_size 4 --no_karzmarz
  else
    printf '%q ' "$PY" -u "$TRAINER" \
      --config "$CONFIG" --env_name "$env" --seed "$seed" --device 0 \
      --timesteps_per_proc "$timesteps" --pi_epochs 4 --lr_pi 0.05 \
      --cg_damping "$DAMPING" --fisher_kernel exact \
      --fisher_kernel_normalization none \
      --actor_curvature_subsample 256 --actor_group_size 4 \
      --critic_update gn --critic_optimizer sgd --critic_curvature_subsample 256
  fi
}

run_one() {
  local gpu=$1 env=$2 seed=$3 timesteps=$4 stdout_path=$5 stderr_path=$6
  local command
  command=$(command_for "$env" "$seed" "$timesteps")
  CUDA_VISIBLE_DEVICES=$gpu bash -lc "$command" > "$stdout_path" 2> "$stderr_path"
}

preflight() {
  local rc0=0 rc1=0
  run_one 0 halfcheetah 900 16384 "$RUN_ROOT/preflight/gpu0.stdout" "$RUN_ROOT/preflight/gpu0.stderr" &
  local p0=$!
  run_one 1 ant 901 16384 "$RUN_ROOT/preflight/gpu1.stdout" "$RUN_ROOT/preflight/gpu1.stderr" &
  local p1=$!
  wait "$p0" || rc0=$?
  wait "$p1" || rc1=$?
  printf "gpu0_rc=%s\ngpu1_rc=%s\n" "$rc0" "$rc1" > "$RUN_ROOT/preflight/status.txt"
  [[ $rc0 -eq 0 && $rc1 -eq 0 ]]
}

run_env() {
  local gpu=$1 env=$2 outdir="$RUN_ROOT/$2"
  mkdir -p "$outdir"
  echo RUNNING > "$outdir/status"
  local pids=() seeds=()
  for seed in 0 1 2 3 4; do
    run_one "$gpu" "$env" "$seed" 10000000 "$outdir/seed${seed}.stdout" "$outdir/seed${seed}.stderr" &
    pids+=("$!")
    seeds+=("$seed")
    echo "$!" > "$outdir/seed${seed}.pid"
    sleep 1
  done
  local failed=0
  for i in "${!pids[@]}"; do
    local rc=0
    wait "${pids[$i]}" || rc=$?
    echo "$rc" > "$outdir/seed${seeds[$i]}.rc"
    [[ $rc -eq 0 ]] || failed=1
  done
  if [[ $failed -eq 0 ]]; then
    echo FINISHED > "$outdir/status"
  else
    echo FAILED > "$outdir/status"
    return 1
  fi
}

run_queue() {
  local gpu=$1
  shift
  local env
  for env in "$@"; do
    run_env "$gpu" "$env"
  done
}

echo RUNNING > "$RUN_ROOT/status"
preflight
run_queue 0 "${ENVS_GPU0[@]}" & worker0=$!
run_queue 1 "${ENVS_GPU1[@]}" & worker1=$!
rc=0
wait "$worker0" || rc=1
wait "$worker1" || rc=1
if [[ $rc -eq 0 ]]; then echo FINISHED > "$RUN_ROOT/status"; else echo FAILED > "$RUN_ROOT/status"; fi
exit "$rc"
