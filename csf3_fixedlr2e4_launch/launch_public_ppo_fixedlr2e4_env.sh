#!/bin/bash
set -euo pipefail

ROOT=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
PY=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
MODE=${MODE:?MODE must be shared or detach}
ENV_NAME=${ENV_NAME:?ENV_NAME must be set}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be set}

case "$MODE" in
  shared)
    TRAINER="$ROOT/train_shared_ppo_public.py"
    CONFIG=ppo_mlp_shared_fixedlr2e4_public.yaml
    ;;
  detach)
    TRAINER="$ROOT/train_detach_ppo_public.py"
    CONFIG=ppo_mlp_detach_fixedlr2e4_public.yaml
    ;;
  *)
    echo "unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

OUTDIR="$RUN_ROOT/$ENV_NAME"
mkdir -p "$OUTDIR"
cd "$ROOT"

export PATH=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cat > "$OUTDIR/run_info.txt" <<EOF
method=public PPO fixed learning rate
network_mode=$MODE
environment=$ENV_NAME
seeds=0,1,2,3,4
timesteps_per_seed=10000000
optimizer=Adam
actor_lr=0.0002
critic_lr=0.0002
kl_adaptive_lr=false
cliprange=0.2
actor_epochs=4
actor_minibatches=8
critic_epochs=4
critic_minibatches=8
with_popart=true
max_grad_norm=0.5
trainer=$TRAINER
config=$ROOT/configs/$CONFIG
trainer_sha256=$(sha256sum "$TRAINER" | awk '{print $1}')
config_sha256=$(sha256sum "$ROOT/configs/$CONFIG" | awk '{print $1}')
slurm_job_id=${SLURM_JOB_ID:-manual}
slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-none}
slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-none}
host=$(hostname)
started_utc=$(date -u +%FT%TZ)
EOF

printf "seed\tstatus\trc\tworkdir\n" > "$OUTDIR/seed_status.tsv"
echo RUNNING > "$OUTDIR/status"

pids=()
seeds=()
for seed in 0 1 2 3 4; do
  workdir="$OUTDIR/seed${seed}_work"
  mkdir -p "$workdir"
  ln -sfn "$ROOT/configs" "$workdir/configs"
  echo RUNNING > "$OUTDIR/seed${seed}.status"
  (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES=0 "$PY" -u "$TRAINER" \
      --config "$CONFIG" \
      --env_name "$ENV_NAME" \
      --seed "$seed" \
      --device 0 \
      --timesteps_per_proc 10000000
  ) > "$OUTDIR/seed${seed}.stdout" 2> "$OUTDIR/seed${seed}.stderr" &
  pids+=("$!")
  seeds+=("$seed")
  echo "$!" > "$OUTDIR/seed${seed}.pid"
  sleep 1
done

failed=0
for i in "${!pids[@]}"; do
  seed=${seeds[$i]}
  rc=0
  wait "${pids[$i]}" || rc=$?
  echo "$rc" > "$OUTDIR/seed${seed}.rc"
  if [[ $rc -eq 0 ]]; then
    echo FINISHED > "$OUTDIR/seed${seed}.status"
    printf "%s\tFINISHED\t0\t%s\n" "$seed" "$OUTDIR/seed${seed}_work" >> "$OUTDIR/seed_status.tsv"
  else
    echo FAILED > "$OUTDIR/seed${seed}.status"
    printf "%s\tFAILED\t%s\t%s\n" "$seed" "$rc" "$OUTDIR/seed${seed}_work" >> "$OUTDIR/seed_status.tsv"
    failed=1
  fi
done

if [[ $failed -eq 0 ]]; then
  echo FINISHED > "$OUTDIR/status"
else
  echo FAILED > "$OUTDIR/status"
  exit 1
fi
