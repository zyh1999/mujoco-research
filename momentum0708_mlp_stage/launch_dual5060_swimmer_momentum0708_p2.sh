#!/usr/bin/env bash
set -Eeuo pipefail

STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack5060}
SOURCE_NAME=${SOURCE_NAME:-momentum0708_mlp_source_20260824}
RUN_REL=${RUN_REL:-perf_runs/dual5060_swimmer_fullEF_fullGGN_momentum0708_s2_10m_p2_20260824}
IMAGE=${IMAGE:-rlstack5060/mujoco-rat-swimmerv3:cu128}
HOST_SOURCE=$STACK_ROOT/workspaces/$SOURCE_NAME
HOST_RUN=$STACK_ROOT/workspaces/$RUN_REL
CONTAINER_SOURCE=/workspace/user/$SOURCE_NAME
CONTAINER_RUN=/workspace/user/$RUN_REL

[[ -f "$HOST_SOURCE/run_full_momentum_cell.py" ]]
[[ -f "$HOST_SOURCE/train_detach_smallbatch_momentum.py" ]]
[[ -f "$HOST_SOURCE/configs/rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml" ]]
[[ ! -e "$HOST_RUN" ]]
mkdir -p "$HOST_RUN"

sha256sum \
  "$HOST_SOURCE/run_full_momentum_cell.py" \
  "$HOST_SOURCE/train_detach_smallbatch_momentum.py" \
  "$HOST_SOURCE/configs/rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml" \
  > "$HOST_RUN/source.sha256"
{
  echo "task_id=MUJOCO-MLP-FULLEF-FULLGGN-MOMENTUM-0708-20260824-04R"
  echo "host=$(hostname -f)"
  echo "image=$IMAGE"
  echo "image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  echo "envs=swimmer"
  echo "momenta=0.7,0.8"
  echo "seeds=0,1"
  echo "placement=2_physical_5060_each_2_trainers"
  echo "started=$(date --iso-8601=seconds)"
} > "$HOST_RUN/run_info.txt"

run_cell() {
  local gpu=$1 momentum=$2 mode=${3:-formal}
  local preflight=()
  [[ "$mode" == preflight ]] && preflight=(--preflight)
  docker run --rm \
    --gpus "device=$gpu" \
    --ipc host \
    -e MUJOCO_GL=glfw \
    -e PYOPENGL_PLATFORM=glfw \
    -e MUJOCO_EGL_DEVICE_ID=0 \
    -e PYTHONUNBUFFERED=1 \
    -v "$STACK_ROOT/workspaces:/workspace/user:rw" \
    -v "$STACK_ROOT/logs/mujoco:/workspace/logs:rw" \
    -w "$CONTAINER_SOURCE" \
    "$IMAGE" \
    python -u "$CONTAINER_SOURCE/run_full_momentum_cell.py" \
      --repo "$CONTAINER_SOURCE" \
      --trainer "$CONTAINER_SOURCE/train_detach_smallbatch_momentum.py" \
      --config rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml \
      --python python \
      --run-root "$CONTAINER_RUN" \
      --env swimmer \
      --momentum "$momentum" \
      --gpu 0 \
      --parallel-seeds \
      --seeds 0 1 \
      "${preflight[@]}"
}

echo PREFLIGHT > "$HOST_RUN/status"
run_cell 0 0.7 preflight > "$HOST_RUN/preflight_m07.launch.log" 2>&1 & p0=$!
run_cell 1 0.8 preflight > "$HOST_RUN/preflight_m08.launch.log" 2>&1 & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
if [[ $rc -ne 0 ]]; then
  echo PREFLIGHT_FAILED > "$HOST_RUN/status"
  exit 1
fi

echo RUNNING > "$HOST_RUN/status"
run_cell 0 0.7 > "$HOST_RUN/formal_m07.launch.log" 2>&1 & q0=$!
run_cell 1 0.8 > "$HOST_RUN/formal_m08.launch.log" 2>&1 & q1=$!
echo "$q0" > "$HOST_RUN/gpu0.worker.pid"
echo "$q1" > "$HOST_RUN/gpu1.worker.pid"
rc=0
wait "$q0" || rc=1
wait "$q1" || rc=1
if [[ $rc -eq 0 ]]; then
  echo FINISHED > "$HOST_RUN/status"
else
  echo FAILED > "$HOST_RUN/status"
fi
date --iso-8601=seconds > "$HOST_RUN/finished_at.txt"
exit "$rc"
