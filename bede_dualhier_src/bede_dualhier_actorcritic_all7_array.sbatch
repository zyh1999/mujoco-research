#!/bin/bash
#SBATCH --job-name=rat_dualhier7
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:00:00
#SBATCH --array=0-6
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.err

set -Eeuo pipefail

ENVS=(ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d)
ENV_NAME=${ENVS[$SLURM_ARRAY_TASK_ID]}
ROOT=/nobackup/projects/bdman37/yihe
RAT_REPO=${ROOT}/src/ICML2026-RAT
PY=${ROOT}/ppc64le/envs/rat/bin/python
DETACH_TRAINER=${RAT_REPO}/train_detach_jointcritic_dualhier16x64.py
SHARED_TRAINER=${RAT_REPO}/train_shared_jointkernel_dualhier32.py
DETACH_CONFIG=rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml
SHARED_CONFIG=rat_mlp_shared.yaml
RUN_ROOT=${ROOT}/perf_runs/bede_dualhier_actorcritic_${ENV_NAME}_d003_all5seed_10m_gpu1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

export TMPDIR=${ROOT}/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PIP_CACHE_DIR=${ROOT}/ppc64le/pip-cache
export XDG_CACHE_HOME=${ROOT}/ppc64le/xdg-cache
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${RUN_ROOT}/no_shared" "${RUN_ROOT}/shared" "${TMPDIR}" "${ROOT}/logs"
module load gcc/12.2
module load cmake/3.18.4
module load cuda/12.4.1
source "${ROOT}/ppc64le/miniconda/etc/profile.d/conda.sh"
conda activate "${ROOT}/ppc64le/envs/rat"

cat > "${RUN_ROOT}/run_info.txt" <<EOF
family=dualhier_actorcritic_16x64_top16_or_top32_aligned1024
platform=bede
hardware_requested=single_gpu
hardware_actual=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
account=bdman37g
partition=${SLURM_JOB_PARTITION}
env=${ENV_NAME}
modes=no_shared_then_shared
seeds=0,1,2,3,4
timesteps_per_seed=10000000
minibatch_samples=1024
local_group_size=64
local_groups_per_block=16
no_shared_actor_top_system=16x16
no_shared_critic_top_system=16x16
shared_actor_local_systems=16x64x64
shared_critic_local_systems=16x64x64
shared_joint_top_system=32x32
shared_cross_terms=preserved_through_shared_parameter_basis
damping=0.03
normalization=none
kaczmarz=false
epochs=4
lr_pi=0.05
lr_v=0.1
critic=GGN
job_id=${SLURM_ARRAY_JOB_ID}
array_task_id=${SLURM_ARRAY_TASK_ID}
host=$(hostname)
created_utc=$(date -u +%FT%TZ)
detach_trainer_sha256=$(sha256sum "${DETACH_TRAINER}" | awk '{print $1}')
shared_trainer_sha256=$(sha256sum "${SHARED_TRAINER}" | awk '{print $1}')
detach_config_sha256=$(sha256sum "${RAT_REPO}/configs/${DETACH_CONFIG}" | awk '{print $1}')
shared_config_sha256=$(sha256sum "${RAT_REPO}/configs/${SHARED_CONFIG}" | awk '{print $1}')
launcher_sha256=$(sha256sum "$0" | awk '{print $1}')
EOF

printf "mode\tenv\tseeds\ttimesteps\tlocal_groups\tgroup_size\ttop_size\tdamping\tnormalization\tkaczmarz\tepochs\tlr_pi\tlr_v\tgpu\n" > "${RUN_ROOT}/manifest.tsv"
printf "no_shared\t%s\t0,1,2,3,4\t10000000\tactor16+critic16\t64\t16+16\t0.03\tnone\tfalse\t4\t0.05\t0.1\t0\n" "${ENV_NAME}" >> "${RUN_ROOT}/manifest.tsv"
printf "shared\t%s\t0,1,2,3,4\t10000000\tactor16+critic16\t64\t32\t0.03\tnone\tfalse\t4\t0.05\t0.1\t0\n" "${ENV_NAME}" >> "${RUN_ROOT}/manifest.tsv"

monitor_gpu() {
  while true; do
    printf '%s,' "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader,nounits
    sleep 60
  done
}
monitor_gpu > "${RUN_ROOT}/gpu_monitor.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null || true' EXIT

run_one() {
  local mode=$1 seed=$2 trainer=$3 config=$4
  local mode_root=${RUN_ROOT}/${mode}
  local workdir=${mode_root}/seed${seed}_work
  mkdir -p "${workdir}"
  ln -sfn "${RAT_REPO}/configs" "${workdir}/configs"
  if [[ ${mode} == no_shared ]]; then
    (
      cd "${workdir}"
      CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${trainer}" \
        --config "${config}" --env_name "${ENV_NAME}" --seed "${seed}" --device 0 \
        --timesteps_per_proc 10000000 --pi_epochs 4 --lr_pi 0.05 \
        --cg_damping 0.03 --fisher_kernel exact \
        --fisher_kernel_normalization none \
        --actor_curvature_subsample 0 --actor_hierarchical_group_size 64 \
        --critic_update gn --critic_optimizer sgd --critic_curvature_subsample 0 \
        --critic_hierarchical_group_size 64
    ) > "${mode_root}/seed${seed}.stdout" 2> "${mode_root}/seed${seed}.stderr"
  else
    (
      cd "${workdir}"
      CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${trainer}" \
        --config "${config}" --env_name "${ENV_NAME}" --seed "${seed}" --device 0 \
        --timesteps_per_proc 10000000 --epochs 4 --lr 0.05 \
        --cg_damping 0.03 --fisher_kernel exact \
        --fisher_kernel_normalization none \
        --actor_curvature_subsample 0 --critic_curvature_subsample 0 \
        --dual_hierarchical_group_size 64 --no_karzmarz
    ) > "${mode_root}/seed${seed}.stdout" 2> "${mode_root}/seed${seed}.stderr"
  fi
}

run_mode() {
  local mode=$1 trainer=$2 config=$3
  local mode_root=${RUN_ROOT}/${mode}
  echo RUNNING > "${mode_root}/status"
  local pids=()
  for seed in 0 1 2 3 4; do
    run_one "${mode}" "${seed}" "${trainer}" "${config}" &
    pids+=("$!")
    echo "$!" > "${mode_root}/seed${seed}.pid"
    sleep 1
  done

  local failed=0
  for seed in 0 1 2 3 4; do
    local rc=0
    wait "${pids[$seed]}" || rc=$?
    echo "${rc}" > "${mode_root}/seed${seed}.rc"
    [[ ${rc} -eq 0 ]] || failed=1
  done
  if [[ ${failed} -eq 0 ]]; then
    echo FINISHED > "${mode_root}/status"
  else
    echo FAILED > "${mode_root}/status"
    return 1
  fi
}

echo RUNNING > "${RUN_ROOT}/status"
if ! run_mode no_shared "${DETACH_TRAINER}" "${DETACH_CONFIG}"; then
  echo FAILED_NO_SHARED > "${RUN_ROOT}/status"
  exit 1
fi
if ! run_mode shared "${SHARED_TRAINER}" "${SHARED_CONFIG}"; then
  echo FAILED_SHARED > "${RUN_ROOT}/status"
  exit 1
fi
echo FINISHED > "${RUN_ROOT}/status"

