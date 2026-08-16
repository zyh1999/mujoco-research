#!/usr/bin/env bash
set -Eeuo pipefail

SRC=/home/yihe/rlstack54/workspaces/transformer_noshared_directionprobe_20260816_v1
ROOT=/home/yihe/rlstack54/workspaces/perf_runs/mujoco_mlp_direction_short_20260816
PYTHON=/home/yihe/.venv/bin/python
ENV_NAME=${1:-ant}
SEED=${2:-17}
GPU=${3:-0}

mkdir -p "$ROOT"
methods=(ppo kfac emp256 energyfree255p1 fullemp)
trainers=(train_detach_ppo_public.py train_detach_ppo_public.py train_detach_jointcritic_actor_fvp_fisherclip_curvsub.py train_detach_energyfree255p1_criticggn256_batch262144.py train_detach_jointcritic_actor_fisherclip.py)
configs=(ppo_mlp_detach_fixedlr_public.yaml kfac_mlp.yaml rat_mlp_detjc_curv256_criticggn_actor_fvpclip05_batch262144.yaml rat_mlp_detjc_energyfree255p1_criticggn256_batch262144_fvpclip05.yaml rat_mlp_detjc_full_ef_ggn_actor_fvp_fisherclip.yaml)

for i in "${!methods[@]}"; do
  method=${methods[$i]}
  out="$ROOT/$method/$ENV_NAME/seed$SEED"
  mkdir -p "$out/work"
  ln -sfn "$SRC/configs" "$out/work/configs"
  (cd "$out/work" && CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=glfw PYOPENGL_PLATFORM=glfw \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" -u "$SRC/${trainers[$i]}" \
      --config "${configs[$i]}" --env_name "$ENV_NAME" --seed "$SEED" --device 0 --timesteps_per_proc 8192 \
      >"$out/stdout.log" 2>"$out/stderr.log") &
  echo "$!" >"$out/pid"
done
wait
