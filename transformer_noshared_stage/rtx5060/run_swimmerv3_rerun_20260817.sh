#!/usr/bin/env bash
set -Eeuo pipefail

# Five-cell-only recovery launcher for MUJOCO-DUAL5060-SWIMMER-RERUN-20260817-03.
# It writes linked rerun records under a new root and never overwrites the
# original dependency-failure artifacts.

STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack5060}
SOURCE_NAME=${SOURCE_NAME:-transformer_noshared_emp_tune_scorefisher_20260815_v2}
SOURCE_COMMIT=${SOURCE_COMMIT:-df9d5e18279d096218a923ad5d4df37c35fdca68}
IMAGE=${MUJOCO_RAT_IMAGE:-rlstack5060/mujoco-rat-swimmerv3:cu128}
ORIGINAL_REL=perf_runs/global7env_selected10m_5060_20260816
RERUN_REL=${RERUN_REL:-perf_runs/global7env_selected10m_5060_20260816_swimmerv3_rerun_20260817}
BASE=$STACK_ROOT/workspaces
SOURCE=$BASE/$SOURCE_NAME
RERUN_ROOT=$BASE/$RERUN_REL

mkdir -p "$RERUN_ROOT/launch" "$RERUN_ROOT/workers"

if find "$RERUN_ROOT" -path '*/swimmer/seed*/status' ! -path '*/preflight/*' -print -quit | grep -q .; then
  echo "refusing duplicate launch: formal status files already exist under $RERUN_ROOT" >&2
  exit 3
fi

cat > "$RERUN_ROOT/launch/immutable_cells.tsv" <<'EOF'
line	method	damping	environment	seed	gpu	trainer	config	original_relative_path
M2	emp256	0.10	swimmer	2	0	train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py	rat_transformer_detjc_emp256_ggn256.yaml	emp256_mom0.5_d0.10_fisher_l2_kl008_10m/emp256/swimmer/seed2
M2	energyfree255p1	0.10	swimmer	1	0	train_detach_energyfree255p1_criticggn256_batch262144.py	rat_transformer_detjc_energyfree255p1_ggn256.yaml	energyfree255p1_mom0.5_d0.10_fisher_l2_kl008_10m/energyfree255p1/swimmer/seed1
M2	energyfree255p1	0.10	swimmer	6	1	train_detach_energyfree255p1_criticggn256_batch262144.py	rat_transformer_detjc_energyfree255p1_ggn256.yaml	energyfree255p1_mom0.5_d0.10_fisher_l2_kl008_10m/energyfree255p1/swimmer/seed6
M2	fullemp	0.10	swimmer	0	0	train_detach_jointcritic_actor_fisherclip.py	rat_transformer_detjc_full_emp_full_ggn.yaml	fullemp_mom0.5_d0.10_fisher_l2_kl008_10m/fullemp/swimmer/seed0
M2	fullemp	0.10	swimmer	5	1	train_detach_jointcritic_actor_fisherclip.py	rat_transformer_detjc_full_emp_full_ggn.yaml	fullemp_mom0.5_d0.10_fisher_l2_kl008_10m/fullemp/swimmer/seed5
EOF

run_cell() {
  local line=$1 method=$2 damping=$3 env_name=$4 seed=$5 gpu=$6 trainer=$7 config=$8 original_rel=$9
  local tag="${method}_mom0.5_d${damping}_fisher_l2_kl008_10m"
  local outdir="$RERUN_ROOT/$tag/$method/$env_name/seed$seed"
  local container_source=/workspace/user/$SOURCE_NAME
  local container_out=/workspace/user/$RERUN_REL/$tag/$method/$env_name/seed$seed
  mkdir -p "$outdir/work"

  {
    echo "task_id=MUJOCO-DUAL5060-SWIMMER-RERUN-20260817-03"
    echo "line=$line"
    echo "method=$method"
    echo "environment=$env_name"
    echo "environment_version=Swimmer-v3"
    echo "seed=$seed"
    echo "physical_gpu=$gpu"
    echo "image=$IMAGE"
    echo "image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')"
    echo "source=$SOURCE"
    echo "source_commit=$SOURCE_COMMIT"
    echo "source_manifest_sha256=$(sha256sum "$SOURCE/source_manifest.sha256" | awk '{print $1}')"
    echo "launcher_sha256=$(sha256sum "$0" | awk '{print $1}')"
    echo "original_failure=$BASE/$ORIGINAL_REL/$original_rel"
    echo "rerun_output=$outdir"
    echo "trainer=$trainer"
    echo "config=$config"
    echo "total_timesteps=10000000"
    echo "expected_logged_steps=9994240"
    echo "rollout_samples=8192"
    echo "actor_epochs=4"
    echo "actor_minibatches=8"
    echo "critic_epochs=4"
    echo "critic_minibatches=8"
    echo "extra_trainer_args=--pi_momentum 0.5 --v_momentum 0.5 --cg_damping $damping --actor_clip_mode fisher_l2 --max_grad_norm 0.5 --kl_target 0.008"
    echo "command=python -u $container_source/$trainer --config $config --env_name $env_name --seed $seed --device 0 --timesteps_per_proc 10000000 --pi_momentum 0.5 --v_momentum 0.5 --cg_damping $damping --actor_clip_mode fisher_l2 --max_grad_norm 0.5 --kl_target 0.008"
    echo "transformer=single_layer_mujoco_body"
    echo "network_mode=no_shared_independent_actor_critic"
    echo "started=$(date --iso-8601=seconds)"
    sha256sum "$SOURCE/$trainer" "$SOURCE/configs/$config" "$SOURCE/utils/mujoco_transformer.py"
  } > "$outdir/run_info.txt"
  echo RUNNING > "$outdir/status"

  set +e
  docker run --rm \
    --gpus "device=$gpu" \
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
    -v "$STACK_ROOT/workspaces:/workspace/user:rw" \
    -v "$STACK_ROOT/logs/mujoco:/workspace/logs:rw" \
    -w /workspace/user \
    "$IMAGE" bash -lc "
      set -Eeuo pipefail
      mkdir -p '$container_out/work'
      ln -sfn '$container_source/configs' '$container_out/work/configs'
      cd '$container_out/work'
      python -u '$container_source/$trainer' \\
        --config '$config' \\
        --env_name '$env_name' \\
        --seed '$seed' \\
        --device 0 \\
        --timesteps_per_proc 10000000 \\
        --pi_momentum 0.5 --v_momentum 0.5 --cg_damping '$damping' \\
        --actor_clip_mode fisher_l2 --max_grad_norm 0.5 --kl_target 0.008
    " > "$outdir/stdout.log" 2> "$outdir/stderr.log"
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]] && grep -Eiq 'out of memory|(^|[^a-z])nan([^a-z]|$)|traceback|assertionerror|linalgerror' "$outdir/stdout.log" "$outdir/stderr.log"; then
    rc=97
    echo 'error marker found in logs' >> "$outdir/stderr.log"
  fi
  if [[ $rc -eq 0 ]] && ! find "$outdir/work/logs" -name progress.csv -type f -size +0c -print -quit | grep -q .; then
    rc=98
    echo 'missing nonempty progress.csv' >> "$outdir/stderr.log"
  fi
  printf '%s\n' "$rc" > "$outdir/rc"
  printf '%s\n' "$(date --iso-8601=seconds)" > "$outdir/finished_at.txt"
  [[ $rc -eq 0 ]] && echo FINISHED > "$outdir/status" || echo FAILED > "$outdir/status"
  return "$rc"
}

declare -a children=()
declare -a labels=()
while IFS=$'\t' read -r line method damping env_name seed gpu trainer config original_rel; do
  [[ $line == line ]] && continue
  run_cell "$line" "$method" "$damping" "$env_name" "$seed" "$gpu" "$trainer" "$config" "$original_rel" \
    > "$RERUN_ROOT/workers/${method}_${env_name}_seed${seed}.launcher.out" \
    2> "$RERUN_ROOT/workers/${method}_${env_name}_seed${seed}.launcher.err" &
  children+=("$!")
  labels+=("${method}_${env_name}_seed${seed}")
done < "$RERUN_ROOT/launch/immutable_cells.tsv"

bundle_rc=0
for i in "${!children[@]}"; do
  if ! wait "${children[$i]}"; then
    echo "${labels[$i]} failed" >&2
    bundle_rc=1
  fi
done
printf '%s\n' "$bundle_rc" > "$RERUN_ROOT/bundle.rc"
exit "$bundle_rc"
