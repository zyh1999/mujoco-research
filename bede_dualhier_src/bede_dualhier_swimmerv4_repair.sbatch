#!/bin/bash
#SBATCH --job-name=rat_swimv4
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:00:00
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%j.err

set -Eeuo pipefail
ROOT=/nobackup/projects/bdman37/yihe
REPO=${ROOT}/src/ICML2026-RAT
PY=${ROOT}/ppc64le/envs/rat/bin/python
DETACH=${REPO}/train_detach_jointcritic_dualhier16x64_swimmerv4.py
SHARED=${REPO}/train_shared_jointkernel_dualhier32_swimmerv4.py
RUN_ROOT=${ROOT}/perf_runs/bede_dualhier_actorcritic_swimmerv4_d003_all5seed_10m_gpu1_repair_${SLURM_JOB_ID}

export TMPDIR=${ROOT}/tmp/${SLURM_JOB_ID}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${RUN_ROOT}/no_shared" "${RUN_ROOT}/shared" "${TMPDIR}" "${ROOT}/logs"
module load gcc/12.2 cmake/3.18.4 cuda/12.4.1
source "${ROOT}/ppc64le/miniconda/etc/profile.d/conda.sh"
conda activate "${ROOT}/ppc64le/envs/rat"

cat > "${RUN_ROOT}/run_info.txt" <<EOF
family=dualhier_actorcritic_swimmerv4_backend_repair
replaces_failed_job=1062170_5
failed_reason=Swimmer-v3_requires_unavailable_mujoco_py_on_ppc64le
compatibility=Swimmer-v4_same_observation_reward_xml_frame_skip_native_mujoco_backend
native_mujoco_version=3.1.6
modes=no_shared_then_shared
seeds=0,1,2,3,4
timesteps_per_seed=10000000
local_systems=actor16x64x64_plus_critic16x64x64
no_shared_top=16x16_plus_16x16
shared_top=32x32
damping=0.03
normalization=none
kaczmarz=false
epochs=4
lr_pi=0.05
lr_v=0.1
host=$(hostname)
job_id=${SLURM_JOB_ID}
detach_sha256=$(sha256sum "${DETACH}" | awk '{print $1}')
shared_sha256=$(sha256sum "${SHARED}" | awk '{print $1}')
EOF

monitor_gpu() {
  while true; do
    printf '%s,' "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    sleep 60
  done
}
monitor_gpu > "${RUN_ROOT}/gpu_monitor.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null || true' EXIT

run_one() {
  local mode=$1 seed=$2 trainer=$3 config=$4
  local mode_root=${RUN_ROOT}/${mode} work=${RUN_ROOT}/${mode}/seed${seed}_work
  mkdir -p "${work}"
  ln -sfn "${REPO}/configs" "${work}/configs"
  if [[ ${mode} == no_shared ]]; then
    (cd "${work}"; CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${trainer}" \
      --config "${config}" --env_name swimmer --seed "${seed}" --device 0 \
      --timesteps_per_proc 10000000 --pi_epochs 4 --lr_pi 0.05 \
      --cg_damping 0.03 --fisher_kernel exact --fisher_kernel_normalization none \
      --actor_curvature_subsample 0 --actor_hierarchical_group_size 64 \
      --critic_update gn --critic_optimizer sgd --critic_curvature_subsample 0 \
      --critic_hierarchical_group_size 64) > "${mode_root}/seed${seed}.stdout" 2> "${mode_root}/seed${seed}.stderr"
  else
    (cd "${work}"; CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${trainer}" \
      --config "${config}" --env_name swimmer --seed "${seed}" --device 0 \
      --timesteps_per_proc 10000000 --epochs 4 --lr 0.05 \
      --cg_damping 0.03 --fisher_kernel exact --fisher_kernel_normalization none \
      --actor_curvature_subsample 0 --critic_curvature_subsample 0 \
      --dual_hierarchical_group_size 64 --no_karzmarz) > "${mode_root}/seed${seed}.stdout" 2> "${mode_root}/seed${seed}.stderr"
  fi
}

run_mode() {
  local mode=$1 trainer=$2 config=$3 mode_root=${RUN_ROOT}/$1
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
  [[ ${failed} -eq 0 ]] || { echo FAILED > "${mode_root}/status"; return 1; }
  echo FINISHED > "${mode_root}/status"
}

echo RUNNING > "${RUN_ROOT}/status"
run_mode no_shared "${DETACH}" rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml || { echo FAILED_NO_SHARED > "${RUN_ROOT}/status"; exit 1; }
run_mode shared "${SHARED}" rat_mlp_shared.yaml || { echo FAILED_SHARED > "${RUN_ROOT}/status"; exit 1; }
echo FINISHED > "${RUN_ROOT}/status"

