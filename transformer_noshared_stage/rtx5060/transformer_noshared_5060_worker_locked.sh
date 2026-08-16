#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU WORKER_ID TASKS_TSV RUN_ROOT SOURCE_NAME" >&2
  exit 2
fi

GPU=$1
WORKER_ID=$2
TASKS_TSV=$3
RUN_ROOT=$4
SOURCE_NAME=$5
STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack5060}
LAUNCH_ROOT=$STACK_ROOT/workspaces/transformer_noshared_5060_launch
HOST_RUN_ROOT=$STACK_ROOT/workspaces/$RUN_ROOT
WORKER_DIR=$HOST_RUN_ROOT/workers/gpu${GPU}_${WORKER_ID}
LOCK_ROOT=$HOST_RUN_ROOT/task_locks
mkdir -p "$WORKER_DIR" "$LOCK_ROOT"

echo RUNNING > "$WORKER_DIR/status"
echo "$$" > "$WORKER_DIR/pid"
{
  echo "gpu=$GPU"
  echo "worker_id=$WORKER_ID"
  echo "tasks=$TASKS_TSV"
  echo "run_root=$HOST_RUN_ROOT"
  echo "source_name=$SOURCE_NAME"
  echo "lock_root=$LOCK_ROOT"
  echo "started=$(date --iso-8601=seconds)"
} > "$WORKER_DIR/run_info.txt"

failed=0
while IFS=$'\t' read -r shard method env_name seed; do
  [[ -n "${shard:-}" ]] || continue
  [[ "$shard" != shard ]] || continue
  [[ "$shard" == "$GPU" ]] || continue

  outdir=$HOST_RUN_ROOT/$method/$env_name/seed${seed}
  status_file=$outdir/status
  task_id=${method}_${env_name}_seed${seed}
  lock_dir=$LOCK_ROOT/$task_id.lock

  if [[ -f "$status_file" ]]; then
    status=$(cat "$status_file")
    if [[ "$status" == FINISHED || "$status" == RUNNING ]]; then
      printf "%s\tSKIP_%s\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "$status" "$method" "$env_name" "$seed" >> "$WORKER_DIR/events.tsv"
      continue
    fi
  fi

  if ! mkdir "$lock_dir" 2>/dev/null; then
    printf "%s\tSKIP_LOCKED\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "$method" "$env_name" "$seed" >> "$WORKER_DIR/events.tsv"
    continue
  fi
  trap 'rm -rf "$lock_dir"; exit 130' INT TERM

  if [[ -f "$status_file" ]]; then
    status=$(cat "$status_file")
    if [[ "$status" == FINISHED || "$status" == RUNNING ]]; then
      printf "%s\tSKIP_%s_AFTER_LOCK\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "$status" "$method" "$env_name" "$seed" >> "$WORKER_DIR/events.tsv"
      rm -rf "$lock_dir"
      trap - INT TERM
      continue
    fi
  fi

  printf "%s\tSTART\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "$method" "$env_name" "$seed" >> "$WORKER_DIR/events.tsv"
  rc=0
  TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000} \
    "$LAUNCH_ROOT/transformer_noshared_5060_run_seed.sh" \
    "$method" "$env_name" "$seed" "$GPU" "$RUN_ROOT" "$SOURCE_NAME" || rc=$?
  printf "%s\tEND\t%s\t%s\t%s\trc=%s\n" "$(date --iso-8601=seconds)" "$method" "$env_name" "$seed" "$rc" >> "$WORKER_DIR/events.tsv"
  rm -rf "$lock_dir"
  trap - INT TERM

  if [[ "$rc" -ne 0 ]]; then
    failed=1
    if [[ "${STOP_ON_FAILURE:-1}" == 1 ]]; then
      echo FAILED > "$WORKER_DIR/status"
      exit "$rc"
    fi
  fi
done < "$TASKS_TSV"

if [[ "$failed" -eq 0 ]]; then
  echo FINISHED > "$WORKER_DIR/status"
else
  echo FAILED > "$WORKER_DIR/status"
  exit 1
fi
