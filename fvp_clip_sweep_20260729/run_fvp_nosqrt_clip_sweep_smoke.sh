#!/bin/bash
set -u

ROOT=/home/yihe/ICML2026-RAT-original-4090
PY=/home/yihe/.venv/bin/python
TRAINER=train_detach_jointcritic_actor_fvp_fisherclip_sweep_20260729.py
SMOKE_ROOT="$ROOT/perf_runs/workstation54_gpu1_fvp_nosqrt_clip_sweep_smoke_retry_20260729"
PHYSICAL_GPU=${PHYSICAL_GPU:-1}

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export LD_LIBRARY_PATH=/home/yihe/.mujoco/mujoco210/bin:/usr/lib/nvidia:/home/yihe/.venv/lib/python3.10/site-packages/cv2/../../lib64
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$SMOKE_ROOT"
cd "$ROOT"

failed=0
for batch in "025 05" "1 2"; do
  pids=()
  labels=()
  for label in $batch; do
    config="rat_mlp_detjc_full_ef_ggn_actor_fvp_nosqrt_clip${label}.yaml"
    out="$SMOKE_ROOT/clip${label}"
    mkdir -p "$out"
    "$PY" -u "$TRAINER" \
      --config "$config" \
      --env_name halfcheetah \
      --seed 900 \
      --device 0 \
      --timesteps_per_proc 100000 \
      --pi_epochs 4 \
      --lr_pi 0.05 \
      --cg_damping 0.03 \
      --fisher_kernel exact \
      --fisher_kernel_normalization none \
      --actor_curvature_subsample 0 \
      --critic_curvature_subsample 0 \
      --no-actor_subsample_full_batch_gradient \
      > "$out/stdout" 2> "$out/stderr" &
    pids+=("$!")
    labels+=("$label")
    echo "$!" > "$out/pid"
    sleep 2
  done

  for i in "${!pids[@]}"; do
    rc=0
    wait "${pids[$i]}" || rc=$?
    label="${labels[$i]}"
    echo "$rc" > "$SMOKE_ROOT/clip${label}/rc"
    if [[ $rc -ne 0 ]]; then
      failed=1
    fi
  done
done

if [[ $failed -eq 0 ]]; then
  echo PASS > "$SMOKE_ROOT/status"
else
  echo FAIL > "$SMOKE_ROOT/status"
  exit 1
fi
