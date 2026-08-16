#!/usr/bin/env bash
set -Eeuo pipefail

root_rel=perf_runs/transformer_noshared_5060_all_methods_all7_5seed_10m_20260813_0600
source_name=transformer_noshared_20260813_v1
stack=$HOME/rlstack5060
run_seed=$stack/workspaces/transformer_noshared_5060_launch/transformer_noshared_5060_run_seed.sh
host_root=$stack/workspaces/$root_rel
method=fullemp
env=halfcheetah
seed=0
gpu=1
out=$host_root/$method/$env/seed$seed
log_root=$host_root/manual_launches
watch_log=$log_root/guarded_fullemp_halfcheetah_seed0_gpu1.watch.log
launch_log=$log_root/fullemp_halfcheetah_seed0_gpu1_guarded_$(date +%Y%m%d_%H%M%S).log
max_wait_sec=${MAX_WAIT_SEC:-172800}
interval=${INTERVAL:-300}
deadline=$(( $(date +%s) + max_wait_sec ))

while true; do
  now=$(date --iso-8601=seconds)

  if [ -f "$out/status" ] && grep -Eq "^(RUNNING|FINISHED)$" "$out/status"; then
    echo "$now skip existing status=$(cat "$out/status")" >> "$watch_log"
    exit 0
  fi

  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d " ")
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d " ")
  proc_count=$(nvidia-smi pmon -c 1 -s um 2>/dev/null | awk -v g="$gpu" '$1==g && /python/ {n++} END{print n+0}')

  echo "$now gpu=$gpu mem=${mem}MiB util=${util}% python_count=$proc_count" >> "$watch_log"

  if [ "$mem" -lt 4500 ] && [ "$util" -lt 65 ] && [ "$proc_count" -le 3 ]; then
    echo "$now launching $method/$env/seed$seed log=$launch_log" >> "$watch_log"
    nohup env TOTAL_TIMESTEPS=10000000 "$run_seed" "$method" "$env" "$seed" "$gpu" "$root_rel" "$source_name" > "$launch_log" 2>&1 &
    echo $! > "$log_root/fullemp_halfcheetah_seed0_gpu1.pid"
    exit 0
  fi

  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "$now timeout without launch" >> "$watch_log"
    exit 124
  fi

  sleep "$interval"
done
