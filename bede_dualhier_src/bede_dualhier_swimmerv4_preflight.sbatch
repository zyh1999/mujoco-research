#!/bin/bash
#SBATCH --job-name=rat_swimv4_pre
#SBATCH --account=bdman37g
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%j.err

set -Eeuo pipefail
ROOT=/nobackup/projects/bdman37/yihe
REPO=${ROOT}/src/ICML2026-RAT
PY=${ROOT}/ppc64le/envs/rat/bin/python
RUN_ROOT=${ROOT}/perf_runs/bede_dualhier_swimmerv4_preflight_${SLURM_JOB_ID}
DETACH=${REPO}/train_detach_jointcritic_dualhier16x64_swimmerv4.py
SHARED=${REPO}/train_shared_jointkernel_dualhier32_swimmerv4.py

export TMPDIR=${ROOT}/tmp/${SLURM_JOB_ID}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${RUN_ROOT}" "${TMPDIR}" "${ROOT}/logs"
trap 'echo FAILED > "${RUN_ROOT}/status"' ERR
module load gcc/12.2 cmake/3.18.4 cuda/12.4.1
source "${ROOT}/ppc64le/miniconda/etc/profile.d/conda.sh"
conda activate "${ROOT}/ppc64le/envs/rat"
echo RUNNING > "${RUN_ROOT}/status"

run_smoke() {
  local mode=$1 trainer=$2 config=$3 seed=$4
  local work=${RUN_ROOT}/${mode}_work
  mkdir -p "${work}"
  ln -sfn "${REPO}/configs" "${work}/configs"
  if [[ ${mode} == no_shared ]]; then
    (cd "${work}"; CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${trainer}" \
      --config "${config}" --env_name swimmer --seed "${seed}" --device 0 \
      --timesteps_per_proc 16384 --pi_epochs 4 --lr_pi 0.05 \
      --cg_damping 0.03 --fisher_kernel exact --fisher_kernel_normalization none \
      --actor_curvature_subsample 0 --actor_hierarchical_group_size 64 \
      --critic_update gn --critic_optimizer sgd --critic_curvature_subsample 0 \
      --critic_hierarchical_group_size 64) \
      > "${RUN_ROOT}/${mode}.stdout" 2> "${RUN_ROOT}/${mode}.stderr"
  else
    (cd "${work}"; CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${trainer}" \
      --config "${config}" --env_name swimmer --seed "${seed}" --device 0 \
      --timesteps_per_proc 16384 --epochs 4 --lr 0.05 \
      --cg_damping 0.03 --fisher_kernel exact --fisher_kernel_normalization none \
      --actor_curvature_subsample 0 --critic_curvature_subsample 0 \
      --dual_hierarchical_group_size 64 --no_karzmarz) \
      > "${RUN_ROOT}/${mode}.stdout" 2> "${RUN_ROOT}/${mode}.stderr"
  fi
}

"${PY}" -m py_compile "${DETACH}" "${SHARED}"
run_smoke no_shared "${DETACH}" rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml 910
run_smoke shared "${SHARED}" rat_mlp_shared.yaml 911
grep -q 'Running on device: cuda:0' "${RUN_ROOT}/no_shared.stdout"
grep -q 'Running on device: cuda:0' "${RUN_ROOT}/shared.stdout"
if grep -Eiq 'OOM|out of memory|NaN|Traceback|LinAlgError' "${RUN_ROOT}"/*.stderr; then
  exit 1
fi
echo FINISHED > "${RUN_ROOT}/status"

