#!/bin/bash
set -u

ROOT=/home/yihe/ICML2026-RAT-original-4090
PY=/home/yihe/.venv/bin/python
TRAINER=train_detach_jointcritic_actor_fvp_fisherclip_sweep_20260729.py
RUN_ROOT=${RUN_ROOT:-$ROOT/perf_runs/workstation54_detjc_fullEF_GGN_actor_fvp_nosqrt_clip_sweep_c025_c05_c1_c2_all7_5seed_10m_20260729}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
PHYSICAL_GPU=${PHYSICAL_GPU:-1}

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export LD_LIBRARY_PATH=/home/yihe/.mujoco/mujoco210/bin:/usr/lib/nvidia:/home/yihe/.venv/lib/python3.10/site-packages/cv2/../../lib64
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$RUN_ROOT"
cd "$ROOT"

cat > "$RUN_ROOT/run_info.txt" <<EOF
family=no_shared_full_ef_full_ggn_actor_fvp_nosqrt_clip_size_sweep
host=mingfei-workstation-31
physical_gpu=$PHYSICAL_GPU
trainer=$TRAINER
clips=0.25,0.5,1.0,2.0
clip_formula=scale_min_1_c_over_q
q=gT_(FVP_plus_dampingI)_g
envs=ant,halfcheetah,hopper,humanoid,humanoidstandup,swimmer,walker2d
seeds=0,1,2,3,4
timesteps=$TOTAL_TIMESTEPS
epochs=4
minibatches=8
minibatch_size=1024
lr_pi=0.05
lr_v=0.1
damping=0.03
actor_curvature_subsample=0
critic_curvature_subsample=0
actor_subsample_full_batch_gradient=false
critic_update=per_sample_GGN
critic_clip=L2_5.0
normalization=none
kaczmarz=false
parallelism=up_to_3_seeds_per_gpu_one_cell_at_a_time
created_utc=$(date -u +%FT%TZ)
EOF

echo RUNNING > "$RUN_ROOT/status"
overall_failed=0

for label in 025 05 1 2; do
  config="rat_mlp_detjc_full_ef_ggn_actor_fvp_nosqrt_clip${label}.yaml"
  for env in ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d; do
    out="$RUN_ROOT/clip${label}/$env"
    mkdir -p "$out"
    echo RUNNING > "$out/status"
    cell_failed=0
    for batch in "0 1 2" "3 4"; do
      pids=()
      seeds=()
      for seed in $batch; do
        if [[ -f "$out/seed${seed}.rc" ]] && [[ "$(cat "$out/seed${seed}.rc")" == 0 ]]; then
          continue
        fi
        "$PY" -u "$TRAINER" \
          --config "$config" \
          --env_name "$env" \
          --seed "$seed" \
          --device 0 \
          --timesteps_per_proc "$TOTAL_TIMESTEPS" \
          --pi_epochs 4 \
          --lr_pi 0.05 \
          --cg_damping 0.03 \
          --fisher_kernel exact \
          --fisher_kernel_normalization none \
          --actor_curvature_subsample 0 \
          --critic_curvature_subsample 0 \
          --no-actor_subsample_full_batch_gradient \
          > "$out/seed${seed}.stdout" 2> "$out/seed${seed}.stderr" &
        pids+=("$!")
        seeds+=("$seed")
        echo "$!" > "$out/seed${seed}.pid"
        sleep 2
      done

      for i in "${!pids[@]}"; do
        rc=0
        wait "${pids[$i]}" || rc=$?
        seed="${seeds[$i]}"
        echo "$rc" > "$out/seed${seed}.rc"
        if [[ $rc -ne 0 ]]; then
          cell_failed=1
          overall_failed=1
        fi
      done
    done

    if [[ $cell_failed -eq 0 ]]; then
      echo FINISHED > "$out/status"
    else
      echo FAILED > "$out/status"
    fi
  done
done

if [[ $overall_failed -eq 0 ]]; then
  echo FINISHED > "$RUN_ROOT/status"
else
  echo PARTIAL_FAILED > "$RUN_ROOT/status"
  exit 1
fi
