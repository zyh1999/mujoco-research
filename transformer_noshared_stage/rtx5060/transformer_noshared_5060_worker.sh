#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU TASKS_TSV RUN_ROOT SOURCE_NAME" >&2
  exit 2
fi

GPU=$1
TASKS_TSV=$2
RUN_ROOT=$3
SOURCE_NAME=$4
STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack5060}
LAUNCH_ROOT=$STACK_ROOT/workspaces/transformer_noshared_5060_launch
HOST_RUN_ROOT=$STACK_ROOT/workspaces/$RUN_ROOT
WORKER_DIR=$HOST_RUN_ROOT/workers/gpu${GPU}
mkdir -p "$WORKER_DIR"

echo RUNNING > "$WORKER_DIR/status"
echo "$$" > "$WORKER_DIR/pid"
{
  echo "gpu=$GPU"
  echo "tasks=$TASKS_TSV"
  echo "run_root=$HOST_RUN_ROOT"
  echo "source_name=$SOURCE_NAME"
  echo "started=$(date --iso-8601=seconds)"
} > "$WORKER_DIR/run_info.txt"

failed=0
while IFS=$'\t' read -r shard method env_name seed; do
  [[ -n "${shard:-}" ]] || continue
  [[ "$shard" != shard ]] || continue
  [[ "$shard" == "$GPU" ]] || continue

  outdir=$HOST_RUN_ROOT/$method/$env_name/seed${seed}
  status_file=$outdir/status
  if [[ -f "$status_file" ]] && [[ "$(cat "$status_file")" == FINISHED ]]; then
    printf "%s\tSKIP_FINISHED\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "$method" "$env_name" "$seed" >> "$WORKER_DIR/events.tsv"
    continue
  fi

  printf "%s\tSTART\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "$method" "$env_name" "$seed" >> "$WORKER_DIR/events.tsv"
  rc=0
  TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000} \
    "$LAUNCH_ROOT/transformer_noshared_5060_run_seed.sh" \
    "$method" "$env_name" "$seed" "$GPU" "$RUN_ROOT" "$SOURCE_NAME" || rc=$?
  printf "%s\tEND\t%s\t%s\t%s\trc=%s\n" "$(date --iso-8601=seconds)" "$method" "$env_name" "$seed" "$rc" >> "$WORKER_DIR/events.tsv"
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
