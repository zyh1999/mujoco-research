#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT SOURCE_NAME" >&2
  exit 2
fi

RUN_ROOT=$1
SOURCE_NAME=$2
STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack5060}
ROOT="$STACK_ROOT/workspaces/$RUN_ROOT"
FIRST="$ROOT/fullemp_mom0.5_d0.03_fisher_norm_retry1/fullemp/hopper/seed0/status"
CONTROL="$ROOT/retry_control"
RUN_SEED="$STACK_ROOT/workspaces/$SOURCE_NAME/rtx5060/transformer_noshared_5060_run_seed.sh"

mkdir -p "$CONTROL"
while [[ ! -f "$FIRST" ]] || [[ "$(<"$FIRST")" == "RUNNING" ]]; do
  sleep 60
done

printf 'First FullEmp retry ended with %s at %s; launching .9/.10.\n' \
  "$(<"$FIRST")" "$(date --iso-8601=seconds)" >> "$CONTROL/fullemp_mom0.9_d0.10_retry1.launcher.log"

export EXTRA_TRAINER_ARGS='--pi_momentum 0.9 --v_momentum 0.9 --cg_damping 0.10 --actor_clip_mode fisher_norm --max_grad_norm 0.5'
exec "$RUN_SEED" fullemp hopper 0 1 \
  "$RUN_ROOT/fullemp_mom0.9_d0.10_fisher_norm_retry1" "$SOURCE_NAME" \
  >> "$CONTROL/fullemp_mom0.9_d0.10_retry1.launcher.log" \
  2>> "$CONTROL/fullemp_mom0.9_d0.10_retry1.launcher.err"
