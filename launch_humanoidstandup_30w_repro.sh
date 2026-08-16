#!/bin/bash
# Reproduce the archived June 4, 2026 no-shared HumanoidStandup seed-0 run.
# The trainer and config are intentionally used in place from the archived source.
#SBATCH -J hs30w_repro
#SBATCH -p gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=125G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/h99859yz/trust-region-main/slurm_logs/%x_%j.out
#SBATCH -e /scratch/h99859yz/trust-region-main/slurm_logs/%x_%j.err
#SBATCH -D /scratch/h99859yz/trust-region-main

set -euo pipefail

SOURCE_ROOT=/scratch/h99859yz/trust-region-main
RUN_STAMP=20260803
RUN_ROOT=/scratch/h99859yz/rat_default_shared_csf3/reruns/humanoidstandup_woodburycg_seed0_originalcfg_10m_${RUN_STAMP}
PYTHON=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TAG=repro30w_video_seed0_originalcfg_10m_${RUN_STAMP}

mkdir -p "${RUN_ROOT}/provenance"
cd "${SOURCE_ROOT}"

cp configs/adv_mlp.yaml "${RUN_ROOT}/provenance/adv_mlp.yaml"
cp "$0" "${RUN_ROOT}/provenance/launcher.sh"
sha256sum train_detach.py utils/utils.py configs/adv_mlp.yaml > "${RUN_ROOT}/provenance/sha256.txt"
{
  printf 'source_root=%s\n' "${SOURCE_ROOT}"
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

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CPATH=/scratch/h99859yz/conda/envs/clean_RL/include/python3.9:${CPATH:-}
export C_INCLUDE_PATH=/scratch/h99859yz/conda/envs/clean_RL/include/python3.9:${C_INCLUDE_PATH:-}
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
touch "${RUN_ROOT}/FINISHED"
