#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/nobackup/projects/bdman37/yihe
REPO=$ROOT/src/ICML2026-RAT
STAGE=$REPO/ktrue_momentum_stage
RUN_ROOT=${RUN_ROOT:-$ROOT/perf_runs/bede_mlp_fullEF_fullGGN_ktrue_m050708_s01_10m_20260825}
[[ ! -e "$RUN_ROOT" ]]
mkdir -p "$RUN_ROOT"
sha256sum \
  "$REPO/momentum0708_mlp_stage/run_full_momentum_cell.py" \
  "$REPO/momentum0708_mlp_stage/train_detach_smallbatch_momentum.py" \
  "$STAGE/rat_mlp_detjc_normnone_ktrue_d003_lrv01_e4_mlp.yaml" \
  "$STAGE/formal_manifest.tsv" > "$RUN_ROOT/source.sha256"
cp "$STAGE/formal_manifest.tsv" "$RUN_ROOT/formal_manifest.tsv"
{
  echo task_id=MUJOCO-MLP-FULLEF-FULLGGN-KTRUE-M050708-S01-20260825-05
  echo host=$(hostname -f)
  echo momenta=0.5,0.7,0.8
  echo envs=ant,halfcheetah,hopper,humanoid,humanoidstandup,swimmer,walker2d
  echo seeds=0,1
  echo placement=wave1_24_then_wave2_18_six_v100_max4_trainers_each
  echo started=$(date --iso-8601=seconds)
} > "$RUN_ROOT/run_info.txt"
preflight=$(sbatch --parsable --export=ALL,RUN_ROOT="$RUN_ROOT" "$STAGE/bede_ktrue_preflight.sbatch")
wave1=$(sbatch --parsable --dependency=afterok:"$preflight" --export=ALL,RUN_ROOT="$RUN_ROOT",WAVE=1 "$STAGE/bede_ktrue_wave.sbatch")
printf 'preflight_job=%s\nwave1_job=%s\n' "$preflight" "$wave1" | tee "$RUN_ROOT/jobs.txt"
