#!/bin/bash
#SBATCH --job-name=rat_dualhier_pre
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
RAT_REPO=${ROOT}/src/ICML2026-RAT
PY=${ROOT}/ppc64le/envs/rat/bin/python
RUN_ROOT=${ROOT}/perf_runs/bede_dualhier_actorcritic_preflight_${SLURM_JOB_ID}
DETACH_TRAINER=${RAT_REPO}/train_detach_jointcritic_dualhier16x64.py
SHARED_TRAINER=${RAT_REPO}/train_shared_jointkernel_dualhier32.py

export TMPDIR=${ROOT}/tmp/${SLURM_JOB_ID}
export PIP_CACHE_DIR=${ROOT}/ppc64le/pip-cache
export XDG_CACHE_HOME=${ROOT}/ppc64le/xdg-cache
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${RUN_ROOT}" "${TMPDIR}" "${ROOT}/logs"
trap 'echo FAILED > "${RUN_ROOT}/status"' ERR
module load gcc/12.2
module load cmake/3.18.4
module load cuda/12.4.1
source "${ROOT}/ppc64le/miniconda/etc/profile.d/conda.sh"
conda activate "${ROOT}/ppc64le/envs/rat"

echo RUNNING > "${RUN_ROOT}/status"
nvidia-smi -L > "${RUN_ROOT}/nvidia_smi_L.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader \
  > "${RUN_ROOT}/nvidia_smi_start.csv"

"${PY}" -m py_compile "${DETACH_TRAINER}" "${SHARED_TRAINER}" \
  "${RAT_REPO}/verify_dualhier_reduced_energy.py"
"${PY}" "${RAT_REPO}/verify_hier16x64_reduced_energy.py" \
  > "${RUN_ROOT}/verify_actor_energy.txt"
"${PY}" "${RAT_REPO}/verify_dualhier_reduced_energy.py" \
  > "${RUN_ROOT}/verify_shared_energy.txt"

run_smoke() {
  local mode=$1 trainer=$2 config=$3
  local workdir=${RUN_ROOT}/${mode}_work
  mkdir -p "${workdir}"
  ln -sfn "${RAT_REPO}/configs" "${workdir}/configs"
  if [[ ${mode} == no_shared ]]; then
    (
      cd "${workdir}"
      CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${trainer}" \
        --config "${config}" --env_name halfcheetah --seed 900 --device 0 \
        --timesteps_per_proc 65536 --pi_epochs 4 --lr_pi 0.05 \
        --cg_damping 0.03 --fisher_kernel exact \
        --fisher_kernel_normalization none \
        --actor_curvature_subsample 0 --actor_hierarchical_group_size 64 \
        --critic_update gn --critic_optimizer sgd --critic_curvature_subsample 0 \
        --critic_hierarchical_group_size 64
    ) > "${RUN_ROOT}/${mode}.stdout" 2> "${RUN_ROOT}/${mode}.stderr"
  else
    (
      cd "${workdir}"
      CUDA_VISIBLE_DEVICES=0 "${PY}" -u "${trainer}" \
        --config "${config}" --env_name halfcheetah --seed 901 --device 0 \
        --timesteps_per_proc 65536 --epochs 4 --lr 0.05 \
        --cg_damping 0.03 --fisher_kernel exact \
        --fisher_kernel_normalization none \
        --actor_curvature_subsample 0 --critic_curvature_subsample 0 \
        --dual_hierarchical_group_size 64 --no_karzmarz
    ) > "${RUN_ROOT}/${mode}.stdout" 2> "${RUN_ROOT}/${mode}.stderr"
  fi
}

run_smoke no_shared "${DETACH_TRAINER}" rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml
NO_SHARED_PROGRESS=$(find "${RUN_ROOT}/no_shared_work" -name progress.csv -type f -print -quit)
test -n "${NO_SHARED_PROGRESS}" && test "$(wc -l < "${NO_SHARED_PROGRESS}")" -ge 2
head -1 "${NO_SHARED_PROGRESS}" | grep -q 'actor_hierarchical_local_relative_residual'
head -1 "${NO_SHARED_PROGRESS}" | grep -q 'critic_hierarchical_local_relative_residual'
echo 0 > "${RUN_ROOT}/no_shared.rc"
run_smoke shared "${SHARED_TRAINER}" rat_mlp_shared.yaml
SHARED_PROGRESS=$(find "${RUN_ROOT}/shared_work" -name progress.csv -type f -print -quit)
test -n "${SHARED_PROGRESS}" && test "$(wc -l < "${SHARED_PROGRESS}")" -ge 2
head -1 "${SHARED_PROGRESS}" | grep -q 'dual_hierarchical_top_size'
head -1 "${SHARED_PROGRESS}" | grep -q 'critic_hierarchical_local_relative_residual'
echo 0 > "${RUN_ROOT}/shared.rc"

for mode in no_shared shared; do
  if grep -Eiq 'OOM|out of memory|NaN|Traceback|LinAlgError' "${RUN_ROOT}/${mode}.stderr"; then
    echo "error marker in ${mode} stderr" >&2
    exit 1
  fi
done
grep -q 'HIER16X64_REDUCED_ENERGY_OK' "${RUN_ROOT}/verify_actor_energy.txt"
grep -q 'DUALHIER16X64_TOP32_REDUCED_ENERGY_OK' "${RUN_ROOT}/verify_shared_energy.txt"
echo FINISHED > "${RUN_ROOT}/status"
