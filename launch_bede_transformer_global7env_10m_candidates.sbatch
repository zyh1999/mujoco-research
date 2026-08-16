#!/bin/bash
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-17
#SBATCH --job-name=tf-g7-10m
#SBATCH --output=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/logs/%x_%A_%a.err

set -Eeuo pipefail

ROOT=/nobackup/projects/bdman37/yihe
SOURCE=$ROOT/src/ICML2026-RAT-transformer-noshared-global7env-v6
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be supplied at submission}
TOTAL_TIMESTEPS=10000000
SEEDS=(0 1 2)
ENVS=(ant halfcheetah hopper humanoid humanoidstandup walker2d)
METHODS=(emp256 energyfree255p1 fullemp)

method_index=$((SLURM_ARRAY_TASK_ID / ${#ENVS[@]}))
env_index=$((SLURM_ARRAY_TASK_ID % ${#ENVS[@]}))
METHOD=${METHODS[$method_index]}
ENV_NAME=${ENVS[$env_index]}

case "$METHOD" in
  emp256)
    DAMPING=0.10
    TRAINER=$SOURCE/train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py
    CONFIG=rat_transformer_detjc_emp256_ggn256.yaml
    MAX_PARALLEL=3
    ;;
  energyfree255p1)
    DAMPING=0.10
    TRAINER=$SOURCE/train_detach_energyfree255p1_criticggn256_batch262144.py
    CONFIG=rat_transformer_detjc_energyfree255p1_ggn256.yaml
    MAX_PARALLEL=3
    ;;
  fullemp)
    DAMPING=0.05
    TRAINER=$SOURCE/train_detach_jointcritic_actor_fisherclip.py
    CONFIG=rat_transformer_detjc_full_emp_full_ggn.yaml
    MAX_PARALLEL=2
    ;;
esac

VARIANT=${METHOD}_mom0.5_d${DAMPING}_fisher_l2_kl008_10m
OUTDIR=$RUN_ROOT/$VARIANT/$METHOD/$ENV_NAME
mkdir -p "$OUTDIR"

module load gcc/12.2
module load cmake/3.18.4
module load cuda/12.4.1
source "$ROOT/ppc64le/miniconda/etc/profile.d/conda.sh"
conda activate "$ROOT/ppc64le/envs/rat"
export TMPDIR=$ROOT/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export LD_LIBRARY_PATH=$ROOT/local/glfw-conda/lib:${LD_LIBRARY_PATH:-}
export PIP_CACHE_DIR=$ROOT/ppc64le/pip-cache
export XDG_CACHE_HOME=$ROOT/ppc64le/xdg-cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$TMPDIR"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "allocated V100 is not visible"
print(torch.cuda.get_device_name(0), flush=True)
PY

{
  echo "identity=global7env_10m_candidate_validation"
  echo "source=$SOURCE"
  echo "method=$METHOD"
  echo "environment=$ENV_NAME"
  echo "damping=$DAMPING"
  echo "seeds=0,1,2"
  echo "timesteps_per_seed=$TOTAL_TIMESTEPS"
  echo "actor_epochs=4"
  echo "actor_minibatches=8"
  echo "critic_epochs=4"
  echo "critic_minibatches=8"
  echo "pi_momentum=0.5"
  echo "v_momentum=0.5"
  echo "actor_clip_mode=fisher_l2"
  echo "max_grad_norm=0.5"
  echo "kl_target=0.008"
  echo "transformer=single_layer_mujoco_body"
  echo "trainer=$TRAINER"
  echo "config=$SOURCE/configs/$CONFIG"
  echo "trainer_sha256=$(sha256sum "$TRAINER" | awk '{print $1}')"
  echo "config_sha256=$(sha256sum "$SOURCE/configs/$CONFIG" | awk '{print $1}')"
  echo "transformer_sha256=$(sha256sum "$SOURCE/utils/mujoco_transformer.py" | awk '{print $1}')"
  echo "job_id=$SLURM_JOB_ID"
  echo "array_task=$SLURM_ARRAY_TASK_ID"
} > "$OUTDIR/run_info.txt"

printf 'seed\tstatus\trc\tworkdir\n' > "$OUTDIR/seed_status.tsv"
echo RUNNING > "$OUTDIR/status"

pids=()
seed_for_pid=()
failed=0

finish_first() {
  local pid=${pids[0]}
  local seed=${seed_for_pid[0]}
  local rc=0
  wait "$pid" || rc=$?
  if [[ $rc -eq 0 ]] && grep -Eiq 'out of memory|(^|[^a-z])nan([^a-z]|$)|traceback|assertionerror|linalgerror' "$OUTDIR/seed${seed}.stdout" "$OUTDIR/seed${seed}.stderr"; then
    rc=97
  fi
  if [[ $rc -eq 0 ]] && ! find "$OUTDIR/seed${seed}_work/logs" -name progress.csv -type f -size +0c -print -quit | grep -q .; then
    rc=98
  fi
  printf '%s\n' "$rc" > "$OUTDIR/seed${seed}.rc"
  if [[ $rc -eq 0 ]]; then
    echo FINISHED > "$OUTDIR/seed${seed}.status"
    printf '%s\tFINISHED\t0\t%s\n' "$seed" "$OUTDIR/seed${seed}_work" >> "$OUTDIR/seed_status.tsv"
  else
    echo FAILED > "$OUTDIR/seed${seed}.status"
    printf '%s\tFAILED\t%s\t%s\n' "$seed" "$rc" "$OUTDIR/seed${seed}_work" >> "$OUTDIR/seed_status.tsv"
    failed=1
  fi
  pids=("${pids[@]:1}")
  seed_for_pid=("${seed_for_pid[@]:1}")
}

for seed in "${SEEDS[@]}"; do
  while [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; do finish_first; done
  workdir=$OUTDIR/seed${seed}_work
  mkdir -p "$workdir"
  ln -sfn "$SOURCE/configs" "$workdir/configs"
  echo RUNNING > "$OUTDIR/seed${seed}.status"
  (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES=0 python -u "$TRAINER" --config "$CONFIG" --env_name "$ENV_NAME" --seed "$seed" --device 0 --timesteps_per_proc "$TOTAL_TIMESTEPS" --pi_momentum 0.5 --v_momentum 0.5 --cg_damping "$DAMPING" --actor_clip_mode fisher_l2 --max_grad_norm 0.5 --kl_target 0.008
  ) > "$OUTDIR/seed${seed}.stdout" 2> "$OUTDIR/seed${seed}.stderr" &
  pids+=("$!")
  seed_for_pid+=("$seed")
done
while [[ ${#pids[@]} -gt 0 ]]; do finish_first; done

if [[ $failed -eq 0 ]]; then echo FINISHED > "$OUTDIR/status"; else echo FAILED > "$OUTDIR/status"; exit 1; fi
