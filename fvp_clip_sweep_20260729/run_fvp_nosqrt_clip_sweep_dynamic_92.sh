#!/bin/bash
set -u

ROOT=/home/yihe/ICML2026-RAT-original-4090
PY=/home/yihe/.venv/bin/python
TRAINER=train_detach_jointcritic_actor_fvp_fisherclip_sweep_20260729.py
RUN_ROOT=${RUN_ROOT:-$ROOT/perf_runs/workstation92_detjc_fullEF_GGN_actor_fvp_nosqrt_clip_sweep_c025_c05_c1_c2_all7_5seed_10m_20260729}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
MIN_FREE_MB=${MIN_FREE_MB:-6500}
MAX_OURS_PER_GPU=${MAX_OURS_PER_GPU:-3}

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
host=mingfei-workstation-92
physical_gpus=0,1
trainer=$TRAINER
trainer_sha256=$(sha256sum "$TRAINER" | awk '{print $1}')
config_clip025_sha256=$(sha256sum configs/rat_mlp_detjc_full_ef_ggn_actor_fvp_nosqrt_clip025.yaml | awk '{print $1}')
config_clip05_sha256=$(sha256sum configs/rat_mlp_detjc_full_ef_ggn_actor_fvp_nosqrt_clip05.yaml | awk '{print $1}')
config_clip1_sha256=$(sha256sum configs/rat_mlp_detjc_full_ef_ggn_actor_fvp_nosqrt_clip1.yaml | awk '{print $1}')
config_clip2_sha256=$(sha256sum configs/rat_mlp_detjc_full_ef_ggn_actor_fvp_nosqrt_clip2.yaml | awk '{print $1}')
launcher_sha256=$(sha256sum "$0" | awk '{print $1}')
preflight=workstation92_fvp_nosqrt_clip_sweep_smoke_20260729_PASS
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
scheduler=memory_aware
min_free_before_launch_mb=$MIN_FREE_MB
max_ours_per_gpu=$MAX_OURS_PER_GPU
created_utc=$(date -u +%FT%TZ)
EOF

echo RUNNING > "$RUN_ROOT/status"

tasks=()
for label in 025 05 1 2; do
  for env in ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d; do
    for seed in 0 1 2 3 4; do
      out="$RUN_ROOT/clip${label}/$env"
      if [[ -f "$out/seed${seed}.rc" ]] && [[ "$(cat "$out/seed${seed}.rc")" == 0 ]]; then
        continue
      fi
      tasks+=("$label:$env:$seed")
    done
  done
done

declare -A pid_gpu
declare -A pid_label
declare -A pid_env
declare -A pid_seed
running=()
next=0
overall_failed=0

gpu_count() {
  local target=$1
  local count=0
  local pid
  for pid in "${running[@]}"; do
    if [[ "${pid_gpu[$pid]}" == "$target" ]] && kill -0 "$pid" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

launch_task() {
  local gpu=$1
  local task=$2
  local label=${task%%:*}
  local rest=${task#*:}
  local env=${rest%%:*}
  local seed=${rest##*:}
  local config="rat_mlp_detjc_full_ef_ggn_actor_fvp_nosqrt_clip${label}.yaml"
  local out="$RUN_ROOT/clip${label}/$env"
  mkdir -p "$out"
  echo RUNNING > "$out/status"

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$TRAINER" \
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
  local pid=$!
  echo "$pid" > "$out/seed${seed}.pid"
  echo "$gpu" > "$out/seed${seed}.gpu"
  pid_gpu[$pid]=$gpu
  pid_label[$pid]=$label
  pid_env[$pid]=$env
  pid_seed[$pid]=$seed
  running+=("$pid")
  printf '%s launch gpu=%s pid=%s clip=%s env=%s seed=%s\n' "$(date -u +%FT%TZ)" "$gpu" "$pid" "$label" "$env" "$seed"
}

while [[ $next -lt ${#tasks[@]} || ${#running[@]} -gt 0 ]]; do
  survivors=()
  for pid in "${running[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      survivors+=("$pid")
      continue
    fi
    rc=0
    wait "$pid" || rc=$?
    label=${pid_label[$pid]}
    env=${pid_env[$pid]}
    seed=${pid_seed[$pid]}
    out="$RUN_ROOT/clip${label}/$env"
    echo "$rc" > "$out/seed${seed}.rc"
    if [[ $rc -ne 0 ]]; then
      overall_failed=1
    fi
    printf '%s finish gpu=%s pid=%s rc=%s clip=%s env=%s seed=%s\n' "$(date -u +%FT%TZ)" "${pid_gpu[$pid]}" "$pid" "$rc" "$label" "$env" "$seed"
    unset 'pid_gpu[$pid]' 'pid_label[$pid]' 'pid_env[$pid]' 'pid_seed[$pid]'
  done
  running=("${survivors[@]}")

  launched=0
  if [[ $next -lt ${#tasks[@]} ]]; then
    for gpu in 0 1; do
      free_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
      ours=$(gpu_count "$gpu")
      if [[ "$free_mb" =~ ^[0-9]+$ ]] && [[ $free_mb -ge $MIN_FREE_MB ]] && [[ $ours -lt $MAX_OURS_PER_GPU ]] && [[ $next -lt ${#tasks[@]} ]]; then
        launch_task "$gpu" "${tasks[$next]}"
        next=$((next + 1))
        launched=1
        sleep 8
      fi
    done
  fi

  if [[ $launched -eq 0 ]]; then
    sleep 30
  fi
done

for label in 025 05 1 2; do
  for env in ant halfcheetah hopper humanoid humanoidstandup swimmer walker2d; do
    out="$RUN_ROOT/clip${label}/$env"
    complete=1
    for seed in 0 1 2 3 4; do
      if [[ ! -f "$out/seed${seed}.rc" ]] || [[ "$(cat "$out/seed${seed}.rc")" != 0 ]]; then
        complete=0
      fi
    done
    if [[ $complete -eq 1 ]]; then
      echo FINISHED > "$out/status"
    else
      echo FAILED > "$out/status"
      overall_failed=1
    fi
  done
done

if [[ $overall_failed -eq 0 ]]; then
  echo FINISHED > "$RUN_ROOT/status"
else
  echo PARTIAL_FAILED > "$RUN_ROOT/status"
  exit 1
fi
