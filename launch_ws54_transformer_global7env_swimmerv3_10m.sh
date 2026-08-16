#!/usr/bin/env bash
set -Eeuo pipefail

STACK_ROOT=${RLSTACK_ROOT:-$HOME/rlstack54}
BASE=$STACK_ROOT/workspaces
SOURCE=transformer_noshared_emp_tune_scorefisher_20260815_v2
RUN_ROOT=perf_runs/transformer_noshared_global7env_10m_swimmerv3_candidates_20260816_v2
ROOT=$BASE/$RUN_ROOT
SEED_RUNNER=$BASE/$SOURCE/rtx5060/run_global7env_native54_swimmerv3_10m_seed.sh
mkdir -p "$ROOT/queues" "$ROOT/workers"

cat > "$ROOT/manifest.txt" <<'EOF'
identity=global7env_10m_candidate_validation
environment=Swimmer-v3
backend=mujoco_py_mujoco210
methods=emp256:d0.10,energyfree255p1:d0.10,fullemp:d0.05
seeds=0,1,2
timesteps_per_seed=10000000
pi_momentum=0.5
v_momentum=0.5
actor_clip_mode=fisher_l2
max_grad_norm=0.5
kl_target=0.008
concurrency=4_per_rtx4090
EOF

printf 'emp256\t0.10\t0\nenergyfree255p1\t0.10\t0\nfullemp\t0.05\t0\nemp256\t0.10\t1\n' > "$ROOT/queues/gpu0.tsv"
printf 'emp256\t0.10\t2\nenergyfree255p1\t0.10\t1\nfullemp\t0.05\t1\nenergyfree255p1\t0.10\t2\nfullemp\t0.05\t2\n' > "$ROOT/queues/gpu1.tsv"

run_worker() {
  local gpu=$1 task=$2 id=$3
  local state=$ROOT/workers/gpu${gpu}_w${id}
  mkdir -p "$state"
  echo RUNNING > "$state/status"
  printf '%s\n' "$$" > "$state/pid"
  while IFS=$'\t' read -r method damping seed; do
    [[ -n "$method" ]] || continue
    "$SEED_RUNNER" "$method" "$seed" "$gpu" "$RUN_ROOT" "$SOURCE" "$damping" || true
  done < "$task"
  echo FINISHED > "$state/status"
}

for gpu in 0 1; do
  task=$ROOT/queues/gpu${gpu}.tsv
  if [[ $gpu -eq 0 ]]; then
    split -l 1 -d -a 1 "$task" "$ROOT/queues/gpu${gpu}_w"
  else
    head -n 2 "$task" > "$ROOT/queues/gpu${gpu}_w0"
    tail -n +3 "$task" | split -l 1 -d -a 1 - "$ROOT/queues/gpu${gpu}_w1"
  fi
  for shard in "$ROOT"/queues/gpu${gpu}_w*; do
    [[ -f "$shard" ]] || continue
    id=${shard##*w}
    run_worker "$gpu" "$shard" "$id" > "$ROOT/workers/gpu${gpu}_w${id}.stdout" 2> "$ROOT/workers/gpu${gpu}_w${id}.stderr" &
  done
done
wait
