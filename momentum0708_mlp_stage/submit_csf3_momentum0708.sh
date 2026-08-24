#!/usr/bin/env bash
set -Eeuo pipefail
REPO=/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original
STAGE=$REPO/momentum0708_mlp_stage_20260824
RUN_ROOT=${RUN_ROOT:-$REPO/perf_runs/csf3_mlp_fullEF_fullGGN_momentum0708_s2_10m_20260824}
[[ ! -e "$RUN_ROOT" ]]
mkdir -p "$RUN_ROOT"
sha256sum "$STAGE/run_full_momentum_cell.py" "$STAGE/train_detach_smallbatch_momentum.py" "$REPO/configs/rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml" > "$RUN_ROOT/source.sha256"
{
  echo "task_id=MUJOCO-MLP-FULLEF-FULLGGN-MOMENTUM-0708-20260824-04R"
  echo "envs=halfcheetah,humanoid,humanoidstandup"
  echo "momenta=0.7,0.8"
  echo "seeds=0,1"
  echo "bede_fallback_reason=Requested node configuration is not available"
  echo "started=$(date --iso-8601=seconds)"
} > "$RUN_ROOT/run_info.txt"
pf=$(sbatch --parsable --export=ALL,RUN_ROOT="$RUN_ROOT" "$STAGE/csf3_momentum0708_preflight.sbatch")
formal=$(sbatch --parsable --dependency=afterok:"$pf" --export=ALL,RUN_ROOT="$RUN_ROOT" "$STAGE/csf3_momentum0708_formal.sbatch")
printf 'preflight_job=%s\nformal_job=%s\n' "$pf" "$formal" | tee "$RUN_ROOT/jobs.txt"
