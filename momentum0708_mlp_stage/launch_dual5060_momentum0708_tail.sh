#!/usr/bin/env bash
set -Eeuo pipefail

STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack5060}
SOURCE_NAME=${SOURCE_NAME:-momentum0708_mlp_source_20260824}
RUN_REL=${RUN_REL:-perf_runs/dual5060_mlp_fullEF_fullGGN_momentum0708_s2_10m_20260824}
IMAGE=${IMAGE:-rlstack5060/mujoco-rat-swimmerv3:cu128}
HOST_SOURCE=$STACK_ROOT/workspaces/$SOURCE_NAME
HOST_RUN=$STACK_ROOT/workspaces/$RUN_REL
CONTAINER_SOURCE=/workspace/user/$SOURCE_NAME
CONTAINER_RUN=/workspace/user/$RUN_REL

[[ -f "$HOST_SOURCE/run_full_momentum_cell.py" ]]
[[ -f "$HOST_SOURCE/train_detach_smallbatch_momentum.py" ]]
[[ -d "$HOST_RUN" ]]
[[ ! -e "$HOST_RUN/tail_started_at.txt" ]]

date --iso-8601=seconds > "$HOST_RUN/tail_queued_at.txt"
echo WAITING_FOR_PRIMARY > "$HOST_RUN/tail_status"
while true; do
  primary_status=$(<"$HOST_RUN/status")
  case "$primary_status" in
    FINISHED) break ;;
    FAILED|PREFLIGHT_FAILED)
      echo PRIMARY_FAILED > "$HOST_RUN/tail_status"
      exit 1
      ;;
  esac
  sleep 60
done

run_cell() {
  local gpu=$1 momentum=$2 env_name=$3
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
      --env "$env_name" \
      --momentum "$momentum" \
      --gpu 0 \
      --seeds 0 1
}

{
  echo "task_id=MUJOCO-MLP-FULLEF-FULLGGN-MOMENTUM-0708-20260824-04R"
  echo "reason=Bede bdman37g scheduler allocation unavailable; user prohibited CSF3"
  echo "envs=halfcheetah,humanoid,humanoidstandup"
  echo "momenta=0.7,0.8"
  echo "seeds=0,1"
  echo "image=$IMAGE"
  echo "image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  echo "started=$(date --iso-8601=seconds)"
} > "$HOST_RUN/tail_run_info.txt"
date --iso-8601=seconds > "$HOST_RUN/tail_started_at.txt"
echo TAIL_RUNNING > "$HOST_RUN/status"
echo RUNNING > "$HOST_RUN/tail_status"

queue0() {
  run_cell 0 0.7 halfcheetah
  run_cell 0 0.8 humanoid
  run_cell 0 0.7 humanoidstandup
}
queue1() {
  run_cell 1 0.8 halfcheetah
  run_cell 1 0.7 humanoid
  run_cell 1 0.8 humanoidstandup
}

queue0 > "$HOST_RUN/gpu0.tail.launch.log" 2>&1 & q0=$!
queue1 > "$HOST_RUN/gpu1.tail.launch.log" 2>&1 & q1=$!
echo "$q0" > "$HOST_RUN/gpu0.tail.worker.pid"
echo "$q1" > "$HOST_RUN/gpu1.tail.worker.pid"
rc=0
wait "$q0" || rc=1
wait "$q1" || rc=1
if [[ $rc -eq 0 ]]; then
  echo FINISHED > "$HOST_RUN/tail_status"
  echo FINISHED > "$HOST_RUN/status"
else
  echo FAILED > "$HOST_RUN/tail_status"
  echo FAILED > "$HOST_RUN/status"
fi
date --iso-8601=seconds > "$HOST_RUN/tail_finished_at.txt"
exit "$rc"
