#!/bin/bash
set -euo pipefail

ROOT=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
PY=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be set}
ENV_NAME=${ENV_NAME:?ENV_NAME must be set}
TRAINER=train_detach_jointcritic_hierarchical_16x64.py
CONFIG=rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml
DAMPING=${DAMPING:-0.03}

case "$ENV_NAME" in
  ant|hopper|humanoid|humanoidstandup|swimmer|walker2d) ;;
  *) echo "unsupported environment: $ENV_NAME" >&2; exit 2 ;;
esac

cd "$ROOT"
mkdir -p "$RUN_ROOT/preflight" "$RUN_ROOT/$ENV_NAME"

export PATH=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cat > "$RUN_ROOT/run_info.txt" <<EOF
family=no_shared_hier16x64_aligned1024_ggn_normnone_damp${DAMPING}_kfalse
mode=no_shared
trainer=${TRAINER}
config=${CONFIG}
actor_samples=1024
actor_groups=16
actor_group_size=64
actor_level1_systems=16x64x64
actor_level1_kernel_denominator=1024
actor_level1_system=H_g_H_gT_div_1024_times_D_ratio_plus_damping_I
actor_level2_system=16x16
actor_level2_basis=sum_group_ratio_times_local_alpha_times_score
actor_level2_damping_metric=sum_group_ratio_times_local_alpha_squared
actor_level2_rhs=sum_group_ratio_times_local_alpha_times_target
actor_parameter_map=B_transpose_rho_div_1024
actor_partition=random_per_update
critic_update=independent_per_sample_GGN
critic_curvature_subsample=256
critic_kernel_denominator=256
envs=${ENV_NAME}
seeds=0,1,2,3,4
timesteps=10000000
epochs=4
lr_pi=0.05
lr_v=0.1
damping=${DAMPING}
normalization=none
kaczmarz=false
job_id=${SLURM_JOB_ID:-manual}
array_job_id=${SLURM_ARRAY_JOB_ID:-none}
array_task_id=${SLURM_ARRAY_TASK_ID:-none}
host=$(hostname)
created_utc=$(date -u +%FT%TZ)
trainer_sha256=$(sha256sum "$TRAINER" | awk '{print $1}')
launcher_sha256=$(sha256sum "$0" | awk '{print $1}')
EOF

printf "mode\tenv\tseeds\ttimesteps\tepochs\tlr_pi\tlr_v\tdamping\tactor_samples\tactor_groups\tgroup_size\tcritic_rows\tnormalization\tkaczmarz\tgpu\n" > "$RUN_ROOT/manifest.tsv"
printf "no_shared\t%s\t0,1,2,3,4\t10000000\t4\t0.05\t0.1\t%s\t1024\t16\t64\t256\tnone\tfalse\t0\n" "$ENV_NAME" "$DAMPING" >> "$RUN_ROOT/manifest.tsv"

run_one() {
  local seed=$1 timesteps=$2 stdout_path=$3 stderr_path=$4
  CUDA_VISIBLE_DEVICES=0 "$PY" -u "$TRAINER" \
    --config "$CONFIG" --env_name "$ENV_NAME" --seed "$seed" --device 0 \
    --timesteps_per_proc "$timesteps" --pi_epochs 4 --lr_pi 0.05 \
    --cg_damping "$DAMPING" --fisher_kernel exact \
    --fisher_kernel_normalization none \
    --actor_curvature_subsample 0 --actor_hierarchical_group_size 64 \
    --critic_update gn --critic_optimizer sgd --critic_curvature_subsample 256 \
    > "$stdout_path" 2> "$stderr_path"
}

echo RUNNING > "$RUN_ROOT/status"
preflight_rc=0
run_one 900 16384 "$RUN_ROOT/preflight/seed900.stdout" "$RUN_ROOT/preflight/seed900.stderr" || preflight_rc=$?
echo "$preflight_rc" > "$RUN_ROOT/preflight/seed900.rc"
if [[ $preflight_rc -ne 0 ]]; then
  echo FAILED_PREFLIGHT > "$RUN_ROOT/status"
  exit "$preflight_rc"
fi
echo PASSED > "$RUN_ROOT/preflight/status"

pids=()
for seed in 0 1 2 3 4; do
  run_one "$seed" 10000000 \
    "$RUN_ROOT/$ENV_NAME/seed${seed}.stdout" \
    "$RUN_ROOT/$ENV_NAME/seed${seed}.stderr" &
  pids+=("$!")
  echo "$!" > "$RUN_ROOT/$ENV_NAME/seed${seed}.pid"
  sleep 1
done

failed=0
for seed in 0 1 2 3 4; do
  rc=0
  wait "${pids[$seed]}" || rc=$?
  echo "$rc" > "$RUN_ROOT/$ENV_NAME/seed${seed}.rc"
  [[ $rc -eq 0 ]] || failed=1
done

if [[ $failed -eq 0 ]]; then
  echo FINISHED > "$RUN_ROOT/status"
else
  echo FAILED > "$RUN_ROOT/status"
  exit 1
fi
