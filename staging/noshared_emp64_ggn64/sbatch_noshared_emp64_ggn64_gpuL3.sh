#!/bin/bash --login
#SBATCH -J ns_emp64_g64
#SBATCH -p gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-2%3
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=120G
#SBATCH -t 2-00:00:00
#SBATCH -o /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.out
#SBATCH -e /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original/slurm_logs/%x_%A_%a.err
#SBATCH -D /scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original

set -euo pipefail

ROOT=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
PY=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
RUN_ROOT="$ROOT/perf_runs/csf3_detjc_emp64_ggn64_normnone_damp003_kfalse_all7_5seed_10m_gpuL3_${SLURM_ARRAY_JOB_ID}"
MAX_PARALLEL=${MAX_PARALLEL:-5}

mkdir -p "$ROOT/slurm_logs" "$RUN_ROOT"
cd "$ROOT"

export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cat > "$RUN_ROOT/run_info_shard${SLURM_ARRAY_TASK_ID}.txt" <<EOF
family=detjc_emp64_ggn64_normnone_damp003_kfalse
mode=no_shared_emp64_ggn64
trainer=train_detach_jointcritic.py
trainer_sha256=$(sha256sum train_detach_jointcritic.py | awk '{print $1}')
worker=run_noshared_emp64_ggn64_worker.py
worker_sha256=$(sha256sum run_noshared_emp64_ggn64_worker.py | awk '{print $1}')
launcher=sbatch_noshared_emp64_ggn64_gpuL3.sh
launcher_sha256=$(sha256sum sbatch_noshared_emp64_ggn64_gpuL3.sh | awk '{print $1}')
config=rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml
config_sha256=$(sha256sum configs/rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml | awk '{print $1}')
job_id=$SLURM_ARRAY_JOB_ID
array_task_id=$SLURM_ARRAY_TASK_ID
envs=ant,halfcheetah,hopper,humanoid,humanoidstandup,swimmer,walker2d
seeds=0,1,2,3,4
timesteps=10000000
epochs=4
minibatches=8
minibatch_size=1024
lr_pi=0.05
lr_v=0.1
damping=0.03
actor_curvature=empirical_fisher
actor_curvature_subsample=64
critic_update=per_sample_GGN
critic_curvature_subsample=64
actor_subsample_full_batch_gradient=false
normalization=none
kaczmarz=false
max_parallel_per_gpu=$MAX_PARALLEL
created_utc=$(date -u +%FT%TZ)
EOF

"$PY" -u "$ROOT/run_noshared_emp64_ggn64_worker.py" \
  --run-root "$RUN_ROOT" \
  --repo "$ROOT" \
  --python "$PY" \
  --shard "$SLURM_ARRAY_TASK_ID" \
  --max-parallel "$MAX_PARALLEL" \
  --timesteps 10000000
