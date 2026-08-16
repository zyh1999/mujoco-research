#!/bin/bash
#SBATCH --job-name=mj-sh-mom-bede
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-41%7
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.err

set -Eeuo pipefail

ROOT=/nobackup/projects/bdman37/yihe
REPO=${ROOT}/src/ICML2026-RAT
PY=${ROOT}/ppc64le/envs/rat/bin/python
GLFW_PREFIX=${ROOT}/ppc64le/runtime-libs/mujoco-glfw-3.4
RUN_ROOT=${ROOT}/perf_runs/bede_shared_smallbatch_momentum05_09_dual255p1_curv256_full_all7_2seed_10m_${SLURM_ARRAY_JOB_ID}
STAGE=${ROOT}/stages/shared_smallbatch_momentum05_09_20260810

export TMPDIR=${ROOT}/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PIP_CACHE_DIR=${ROOT}/ppc64le/pip-cache
export XDG_CACHE_HOME=${ROOT}/ppc64le/xdg-cache
export LD_LIBRARY_PATH="${GLFW_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYGLFW_LIBRARY="${GLFW_PREFIX}/lib/libglfw.so.3"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${RUN_ROOT}" "${TMPDIR}" "${ROOT}/logs"
module load gcc/12.2
module load cmake/3.18.4
module load cuda/12.4.1
source "${ROOT}/ppc64le/miniconda/etc/profile.d/conda.sh"
conda activate "${ROOT}/ppc64le/envs/rat"

if [[ ! -f "${RUN_ROOT}/run_info.txt" ]]; then
  cat > "${RUN_ROOT}/run_info.txt" <<EOF
family=shared_smallbatch_classic_momentum_matrix
platform=bede
account=bdman37g
partition=gpu
architecture=shared_actor_critic_single_network_single_sgd
methods=dual255p1_independent_actor_critic_anchors,ordinary_curv256,full_curvature
momenta=0.5,0.9
envs=ant,halfcheetah,hopper,humanoid,humanoidstandup,swimmer,walker2d
seeds=0,1
timesteps=10000000
rollout=32x256=8192
minibatches=8
minibatch_size=1024
epochs=4
lr=0.05
vf_coef=4.0_rhs_only
damping=0.03
fisher_kernel=exact
fisher_kernel_normalization=none
kaczmarz=false
l2_clip=0.5
array_job_id=${SLURM_ARRAY_JOB_ID}
created_utc=$(date -u +%FT%TZ)
EOF
fi

EXTRA_ARGS=()
if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke-only)
fi

"${PY}" -u "${STAGE}/run_shared_smallbatch_momentum_worker.py" \
  --run-root "${RUN_ROOT}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --python "${PY}" \
  --repo "${REPO}" \
  --trainer-root "${STAGE}" \
  "${EXTRA_ARGS[@]}"
