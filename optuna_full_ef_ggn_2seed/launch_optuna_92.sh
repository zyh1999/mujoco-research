#!/bin/bash
set -euo pipefail

REPO=/home/yihe/ICML2026-RAT-original-4090
ROOT=$REPO/perf_runs/workstation92_detjc_fullEF_fullGGN_clip_momentum_optuna_2seed_d003_20260805
PY=/home/yihe/.venv/bin/python

export LD_LIBRARY_PATH=/home/yihe/.mujoco/mujoco210/bin:/usr/lib/nvidia:/home/yihe/.venv/lib/python3.10/site-packages/cv2/../../lib64
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$ROOT"
cd "$REPO"
cp optuna_full_ef_ggn_2seed/anchor_objectives.json "$ROOT/anchor_objectives.json"

cat > "$ROOT/run_info.txt" <<EOF
family=no_shared_fullEF_fullGGN_clip_momentum_optuna
host=mingfei-workstation-92
gpus=0,1
objective=per_environment_mean_of_seed0_seed1_10M_last10_eprewmean
imported_anchors=4
new_trials_per_environment=8
clip_modes=l2,fvp_fisher_nosqrt,fvp_fisher_sqrt
clip_radii=0.125,0.25,0.5,1.0,2.0,4.0
shared_actor_critic_momenta=0.0,0.5,0.9
damping=0.03
kaczmarz=false
fisher_kernel_normalization=none
actor_curvature_rows=1024
critic_curvature_rows=1024
rollout=32x256
minibatches=8
epochs=4
lr_pi=0.05_adaptive
lr_v=0.1
critic_clip=parameter_L2_5.0
seeds=0,1
timesteps=10000000
trainer_sha256=$(sha256sum train_detach_jointcritic_clip_momentum_optuna.py | awk '{print $1}')
config_sha256=$(sha256sum configs/rat_mlp_detjc_full_ef_ggn_optuna.yaml | awk '{print $1}')
worker_sha256=$(sha256sum optuna_full_ef_ggn_2seed/run_optuna_worker.py | awk '{print $1}')
launcher_sha256=$(sha256sum "$0" | awk '{print $1}')
created_utc=$(date -u +%FT%TZ)
EOF

echo RUNNING > "$ROOT/status"
nohup "$PY" -u optuna_full_ef_ggn_2seed/run_optuna_worker.py \
  --gpu 0 --envs ant hopper humanoid swimmer \
  --repo "$REPO" --root "$ROOT" --new-trials 8 --min-free-mb 8000 \
  > "$ROOT/worker_gpu0.stdout" 2> "$ROOT/worker_gpu0.stderr" &
echo $! > "$ROOT/worker_gpu0.pid"

nohup "$PY" -u optuna_full_ef_ggn_2seed/run_optuna_worker.py \
  --gpu 1 --envs halfcheetah humanoidstandup walker2d \
  --repo "$REPO" --root "$ROOT" --new-trials 8 --min-free-mb 8000 \
  > "$ROOT/worker_gpu1.stdout" 2> "$ROOT/worker_gpu1.stderr" &
echo $! > "$ROOT/worker_gpu1.pid"

echo "gpu0_pid=$(cat "$ROOT/worker_gpu0.pid") gpu1_pid=$(cat "$ROOT/worker_gpu1.pid")"
