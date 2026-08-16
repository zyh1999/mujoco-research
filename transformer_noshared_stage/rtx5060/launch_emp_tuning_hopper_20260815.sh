#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT SOURCE_NAME" >&2
  exit 2
fi

RUN_ROOT=$1
SOURCE_NAME=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_SEED="$SCRIPT_DIR/transformer_noshared_5060_run_seed.sh"
RUN_ROOT_ABS=${RLSTACK_ROOT:-$HOME/rlstack5060}/workspaces/$RUN_ROOT

mkdir -p "$RUN_ROOT_ABS"
printf 'method\tmomentum\tdamping\tclip\tgpu\n' > "$RUN_ROOT_ABS/manifest.tsv"

# One seed per candidate is a screening pass. All candidates retain adaptive KL,
# ratio-in-Gram, Kaczmarz=false, and the same 1024x8/e4 transformer setup.
TASKS=(
  'emp256 0.5 0.03 fisher_norm 0'
  'emp256 0.9 0.03 fisher_norm 0'
  'emp256 0.9 0.10 fisher_norm 0'
  'energyfree255p1 0.5 0.03 fisher_norm 0'
  'energyfree255p1 0.9 0.03 fisher_norm 1'
  'energyfree255p1 0.9 0.10 fisher_norm 1'
  'fullemp 0.5 0.03 fisher_norm 1'
  'fullemp 0.9 0.10 fisher_norm 1'
)

pids=()
for task in "${TASKS[@]}"; do
  read -r method momentum damping clip gpu <<< "$task"
  tag="${method}_mom${momentum}_d${damping}_${clip}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$method" "$momentum" "$damping" "$clip" "$gpu" >> "$RUN_ROOT_ABS/manifest.tsv"
  (
    export EXTRA_TRAINER_ARGS="--pi_momentum $momentum --v_momentum $momentum --cg_damping $damping --actor_clip_mode $clip --max_grad_norm 0.5"
    exec "$RUN_SEED" "$method" hopper 0 "$gpu" "$RUN_ROOT/$tag" "$SOURCE_NAME"
  ) > "$RUN_ROOT_ABS/${tag}.launcher.out" 2> "$RUN_ROOT_ABS/${tag}.launcher.err" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
exit "$rc"
