#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 METHOD SEED GPU RUN_ROOT SOURCE_NAME DAMPING" >&2
  exit 2
fi

METHOD=$1
SEED=$2
GPU=$3
RUN_ROOT=$4
SOURCE_NAME=$5
DAMPING=$6
STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack54}
SOURCE=$STACK_ROOT/workspaces/$SOURCE_NAME
OUTDIR=$STACK_ROOT/workspaces/$RUN_ROOT/${METHOD}_mom0.5_d${DAMPING}_fisher_l2_kl008_10m/$METHOD/swimmer/seed${SEED}

case "$METHOD" in
  emp256) TRAINER=train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py; CONFIG=rat_transformer_detjc_emp256_ggn256.yaml ;;
  energyfree255p1) TRAINER=train_detach_energyfree255p1_criticggn256_batch262144.py; CONFIG=rat_transformer_detjc_energyfree255p1_ggn256.yaml ;;
  fullemp) TRAINER=train_detach_jointcritic_actor_fisherclip.py; CONFIG=rat_transformer_detjc_full_emp_full_ggn.yaml ;;
  *) echo "unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

mkdir -p "$OUTDIR/work"
ln -sfn "$SOURCE/configs" "$OUTDIR/work/configs"
{
  echo "identity=global7env_10m_candidate_validation"
  echo "method=$METHOD"
  echo "environment=Swimmer-v3"
  echo "seed=$SEED"
  echo "physical_gpu=$GPU"
  echo "backend=mujoco_py_mujoco210"
  echo "damping=$DAMPING"
  echo "total_timesteps=10000000"
  echo "pi_momentum=0.5"
  echo "v_momentum=0.5"
  echo "actor_clip_mode=fisher_l2"
  echo "max_grad_norm=0.5"
  echo "kl_target=0.008"
  echo "transformer=single_layer_mujoco_body"
  sha256sum "$SOURCE/$TRAINER" "$SOURCE/configs/$CONFIG" "$SOURCE/utils/mujoco_transformer.py"
} > "$OUTDIR/run_info.txt"
echo RUNNING > "$OUTDIR/status"

set +e
(
  cd "$OUTDIR/work"
  CUDA_VISIBLE_DEVICES="$GPU" \
  LD_LIBRARY_PATH=/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/home/yihe/.mujoco/mujoco210/bin \
  MUJOCO_PY_MUJOCO_PATH=/home/yihe/.mujoco/mujoco210 \
  SB3_LEGACY_MUJOCO_PY=1 \
  PYTHONPATH=/home/yihe/rlstack54/workspaces/sb3_swimmerv3_compat_20260815:${PYTHONPATH:-} \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/yihe/.venv/bin/python -u "$SOURCE/$TRAINER" \
    --config "$CONFIG" --env_name swimmer --seed "$SEED" --device 0 \
    --timesteps_per_proc 10000000 \
    --pi_momentum 0.5 --v_momentum 0.5 --cg_damping "$DAMPING" \
    --actor_clip_mode fisher_l2 --max_grad_norm 0.5 --kl_target 0.008
) > "$OUTDIR/stdout.log" 2> "$OUTDIR/stderr.log"
rc=$?
set -e
if [[ $rc -eq 0 ]] && grep -Eiq 'out of memory|(^|[^a-z])nan([^a-z]|$)|traceback|assertionerror|linalgerror' "$OUTDIR/stdout.log" "$OUTDIR/stderr.log"; then rc=97; fi
if [[ $rc -eq 0 ]] && ! find "$OUTDIR/work/logs" -name progress.csv -type f -size +0c -print -quit | grep -q .; then rc=98; fi
printf '%s\n' "$rc" > "$OUTDIR/rc"
[[ $rc -eq 0 ]] && echo FINISHED > "$OUTDIR/status" || echo FAILED > "$OUTDIR/status"
exit "$rc"
