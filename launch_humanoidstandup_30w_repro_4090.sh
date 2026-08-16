#!/bin/bash
# Original-config HumanoidStandup seed-0 reproduction on workstation 10.49.7.54.
set -euo pipefail

SOURCE_ROOT=/home/yihe/humanoidstandup_legacy_30w_source_20260604
RUN_ROOT=${SOURCE_ROOT}/runs/hs30w_seed0_originalcfg_10m_20260803
PYTHON=/home/yihe/.venv/bin/python
TAG=hs30w_seed0_originalcfg_10m_20260803

mkdir -p "${RUN_ROOT}/provenance"
cd "${SOURCE_ROOT}"

cp configs/adv_mlp.yaml "${RUN_ROOT}/provenance/adv_mlp.yaml"
cp "$0" "${RUN_ROOT}/provenance/launcher.sh"
sha256sum train_detach.py utils/utils.py configs/adv_mlp.yaml > "${RUN_ROOT}/provenance/sha256.txt"
{
  printf 'host=%s\n' "$(hostname)"
  printf 'gpu=0\n'
  printf 'seed=0\n'
  printf 'env=humanoidstandup\n'
  printf 'timesteps=10000000\n'
  printf 'fisher_kernel=exact\n'
  printf 'advantage_transform=woodbury_cg\n'
  printf 'cg_steps=40\n'
  printf 'fisher_kernel_normalization=none\n'
  printf 'vec_env_type=subproc\n'
  printf 'vec_env_start_method=forkserver\n'
  printf 'rat_log_tag=%s\n' "${TAG}"
} > "${RUN_ROOT}/provenance/run_identity.txt"
nvidia-smi > "${RUN_ROOT}/provenance/nvidia_smi_start.txt"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAT_LOG_TAG="${TAG}"

"${PYTHON}" -u train_detach.py \
  --config adv_mlp.yaml \
  --env_name humanoidstandup \
  --timesteps_per_proc 10000000 \
  --seed 0 \
  --device 0 \
  --fisher_kernel exact \
  --advantage_transform woodbury_cg \
  --cg_steps 40 \
  --fisher_kernel_normalization none \
  --vec_env_type subproc \
  --vec_env_start_method forkserver

LOG_DIR=$(find logs -maxdepth 2 -type d -name "humanoidstandup.*.${TAG}" -print -quit)
test -n "${LOG_DIR}"
test -f "${LOG_DIR}/model.ckpt"
ln -sfn "${SOURCE_ROOT}/${LOG_DIR}/model.ckpt" "${RUN_ROOT}/model.ckpt"
printf '%s\n' "${SOURCE_ROOT}/${LOG_DIR}" > "${RUN_ROOT}/trainer_log_dir.txt"
cp "${LOG_DIR}/config.yaml" "${RUN_ROOT}/provenance/trainer_config.yaml"
nvidia-smi > "${RUN_ROOT}/provenance/nvidia_smi_end.txt"
touch "${RUN_ROOT}/FINISHED"
