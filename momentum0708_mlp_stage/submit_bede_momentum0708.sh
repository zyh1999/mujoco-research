#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/nobackup/projects/bdman37/yihe
STAGE=$ROOT/src/ICML2026-RAT/momentum0708_mlp_stage
RUN_ROOT=${RUN_ROOT:-$ROOT/perf_runs/bede_mlp_fullEF_fullGGN_momentum0708_s2_10m_20260824}
[[ ! -e "$RUN_ROOT" ]]
mkdir -p "$RUN_ROOT"
sha256sum \
  "$STAGE/run_full_momentum_cell.py" \
  "$STAGE/train_detach_smallbatch_momentum.py" \
  "$ROOT/src/ICML2026-RAT/configs/rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml" \
  > "$RUN_ROOT/source.sha256"
{
  echo "task_id=MUJOCO-MLP-FULLEF-FULLGGN-MOMENTUM-0708-20260824-04R"
  echo "host=$(hostname -f)"
  echo "envs=ant,halfcheetah,hopper,humanoid,humanoidstandup,walker2d"
  echo "momenta=0.7,0.8"
  echo "seeds=0,1"
  echo "placement=6_physical_v100_each_4_trainers"
  echo "started=$(date --iso-8601=seconds)"
} > "$RUN_ROOT/run_info.txt"

preflight_job=$(sbatch --parsable --export=ALL,RUN_ROOT="$RUN_ROOT" "$STAGE/bede_momentum0708_preflight.sbatch")
formal_job=$(sbatch --parsable --dependency=afterok:"$preflight_job" --export=ALL,RUN_ROOT="$RUN_ROOT" "$STAGE/bede_momentum0708_formal.sbatch")
printf 'preflight_job=%s\nformal_job=%s\n' "$preflight_job" "$formal_job" | tee "$RUN_ROOT/jobs.txt"
