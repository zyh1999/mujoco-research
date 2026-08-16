#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 METHOD ENV_NAME SEED GPU RUN_ROOT SOURCE_NAME" >&2
  exit 2
fi

METHOD=$1
ENV_NAME=$2
SEED=$3
GPU=$4
RUN_ROOT=$5
SOURCE_NAME=$6
EXTRA_TRAINER_ARGS=${EXTRA_TRAINER_ARGS:-}

STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack5060}
HOST_SOURCE=$STACK_ROOT/workspaces/$SOURCE_NAME
HOST_RUN_ROOT=$STACK_ROOT/workspaces/$RUN_ROOT
CONTAINER_SOURCE=/workspace/user/$SOURCE_NAME
CONTAINER_RUN_ROOT=/workspace/user/$RUN_ROOT

case "$METHOD" in
  ppo)
    TRAINER=train_detach_ppo_public.py
    CONFIG=ppo_transformer_detach_fixedlr3e4_public.yaml
    ;;
  kfac)
    TRAINER=train_detach_ppo_public.py
    CONFIG=kfac_transformer_detach.yaml
    ;;
  emp256)
    TRAINER=train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py
    CONFIG=rat_transformer_detjc_emp256_ggn256.yaml
    ;;
  energyfree255p1)
    TRAINER=train_detach_energyfree255p1_criticggn256_batch262144.py
    CONFIG=rat_transformer_detjc_energyfree255p1_ggn256.yaml
    ;;
  fullemp)
    TRAINER=train_detach_jointcritic_actor_fisherclip.py
    CONFIG=rat_transformer_detjc_full_emp_full_ggn.yaml
    ;;
  *)
    echo "unsupported METHOD=$METHOD" >&2
    exit 2
    ;;
esac

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
TASK_ID=${METHOD}_${ENV_NAME}_seed${SEED}
HOST_OUTDIR=$HOST_RUN_ROOT/$METHOD/$ENV_NAME/seed${SEED}
CONTAINER_OUTDIR=$CONTAINER_RUN_ROOT/$METHOD/$ENV_NAME/seed${SEED}

mkdir -p "$HOST_OUTDIR"
{
  echo "method=$METHOD"
  echo "environment=$ENV_NAME"
  echo "seed=$SEED"
  echo "physical_gpu=$GPU"
  echo "host=$(hostname -f)"
  echo "source=$HOST_SOURCE"
  echo "run_root=$HOST_RUN_ROOT"
  echo "trainer=$TRAINER"
  echo "config=$CONFIG"
  echo "total_timesteps=$TOTAL_TIMESTEPS"
  echo "rollout_samples=8192"
  echo "actor_epochs=4"
  echo "actor_minibatches=8"
  echo "critic_epochs=4"
  echo "critic_minibatches=8"
  echo "extra_trainer_args=$EXTRA_TRAINER_ARGS"
  echo "transformer=single_layer_mujoco_body"
  echo "network_mode=no_shared_independent_actor_critic"
  echo "started=$(date --iso-8601=seconds)"
  sha256sum "$HOST_SOURCE/$TRAINER" "$HOST_SOURCE/configs/$CONFIG" "$HOST_SOURCE/utils/mujoco_transformer.py"
} > "$HOST_OUTDIR/run_info.txt"

echo RUNNING > "$HOST_OUTDIR/status"

set +e
docker run --rm \
  --gpus "device=$GPU" \
  --ipc host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e MUJOCO_GL=glfw \
  -e PYOPENGL_PLATFORM=glfw \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  -e EXTRA_TRAINER_ARGS="$EXTRA_TRAINER_ARGS" \
  -v "$STACK_ROOT/workspaces:/workspace/user:rw" \
  -v "$STACK_ROOT/logs/mujoco:/workspace/logs:rw" \
  -w /workspace/user \
  rlstack5060/mujoco-rat:cu128 bash -lc "
  set -Eeuo pipefail
  export MUJOCO_GL=glfw
  export PYOPENGL_PLATFORM=glfw
  export MUJOCO_EGL_DEVICE_ID=0
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  mkdir -p '$CONTAINER_OUTDIR/work'
  ln -sfn '$CONTAINER_SOURCE/configs' '$CONTAINER_OUTDIR/work/configs'
  cd '$CONTAINER_OUTDIR/work'
  python -u '$CONTAINER_SOURCE/$TRAINER' \
    --config '$CONFIG' \
    --env_name '$ENV_NAME' \
    --seed '$SEED' \
    --device 0 \
    --timesteps_per_proc '$TOTAL_TIMESTEPS' \
    \${EXTRA_TRAINER_ARGS}
" > "$HOST_OUTDIR/stdout.log" 2> "$HOST_OUTDIR/stderr.log"
rc=$?
set -e
echo "$rc" > "$HOST_OUTDIR/rc"
if [[ "$rc" -eq 0 ]] && grep -Eiq "out of memory|(^|[^a-z])nan([^a-z]|$)|traceback|linalgerror" \
  "$HOST_OUTDIR/stdout.log" "$HOST_OUTDIR/stderr.log"; then
  echo "error marker found in logs" >> "$HOST_OUTDIR/stderr.log"
  rc=97
  echo "$rc" > "$HOST_OUTDIR/rc"
fi

if [[ "$rc" -eq 0 ]] && [[ "${ALLOW_EMPTY_PROGRESS:-0}" != 1 ]] && ! find "$HOST_OUTDIR/work/logs" -name progress.csv -type f -size +0c -print -quit | grep -q .; then
  echo "missing nonempty progress.csv" >> "$HOST_OUTDIR/stderr.log"
  rc=98
  echo "$rc" > "$HOST_OUTDIR/rc"
fi

if [[ "$rc" -eq 0 ]]; then
  echo FINISHED > "$HOST_OUTDIR/status"
else
  echo FAILED > "$HOST_OUTDIR/status"
fi
exit "$rc"
