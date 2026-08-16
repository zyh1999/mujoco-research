#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 RUN_ROOT SOURCE METHOD GPU TRIGGER_TAG NEW_TAG TOTAL_TIMESTEPS EXTRA_ARGS" >&2
  exit 2
fi

RUN_ROOT=$1
SOURCE_NAME=$2
METHOD=$3
GPU=$4
TRIGGER_TAG=$5
NEW_TAG=$6
TOTAL_TIMESTEPS=$7
EXTRA_TRAINER_ARGS=$8

STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack5060}
ROOT="$STACK_ROOT/workspaces/$RUN_ROOT"
TRIGGER="$ROOT/$TRIGGER_TAG/$METHOD/hopper/seed0/status"
CONTROL="$ROOT/retry_control"
RUN_SEED="$STACK_ROOT/workspaces/$SOURCE_NAME/rtx5060/transformer_noshared_5060_run_seed.sh"
mkdir -p "$CONTROL"

while [[ ! -f "$TRIGGER" ]] || [[ "$(<"$TRIGGER")" == "RUNNING" ]]; do
  sleep 60
done

printf 'Trigger %s ended (%s) at %s; launching %s.\n' \
  "$TRIGGER_TAG" "$(<"$TRIGGER")" "$(date --iso-8601=seconds)" "$NEW_TAG" \
  >> "$CONTROL/${NEW_TAG}.launcher.log"

export TOTAL_TIMESTEPS EXTRA_TRAINER_ARGS
exec "$RUN_SEED" "$METHOD" hopper 0 "$GPU" "$RUN_ROOT/$NEW_TAG" "$SOURCE_NAME" \
  >> "$CONTROL/${NEW_TAG}.launcher.log" \
  2>> "$CONTROL/${NEW_TAG}.launcher.err"
