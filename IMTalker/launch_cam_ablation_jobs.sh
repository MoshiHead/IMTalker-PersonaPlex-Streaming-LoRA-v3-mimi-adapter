#!/usr/bin/env bash
set -euo pipefail
cd /workspace/imtalker_static/IMTalker
mkdir -p /workspace/imtalker_static/logs /workspace/exps

# Stop only this experiment family.
pids=$(pgrep -f "cam_ablation_remove_1h|cam_ablation_zero_embedding_1h|preview_full_checkpoint" || true)
if [[ -n "$pids" ]]; then
  kill $pids 2>/dev/null || true
  sleep 3
  pids2=$(pgrep -f "cam_ablation_remove_1h|cam_ablation_zero_embedding_1h|preview_full_checkpoint" || true)
  if [[ -n "$pids2" ]]; then
    kill -9 $pids2 2>/dev/null || true
  fi
fi

rm -rf /workspace/exps/cam_ablation_remove_1h /workspace/exps/cam_ablation_zero_embedding_1h

nohup ./run_cam_ablation_1h.sh remove cam_ablation_remove_1h 0,1 12910 \
  > /workspace/imtalker_static/logs/cam_ablation_remove_1h.log 2>&1 < /dev/null &
echo $! > /workspace/imtalker_static/logs/cam_ablation_remove_1h.pid

nohup ./run_cam_ablation_1h.sh zero_embedding cam_ablation_zero_embedding_1h 2,3 12920 \
  > /workspace/imtalker_static/logs/cam_ablation_zero_embedding_1h.log 2>&1 < /dev/null &
echo $! > /workspace/imtalker_static/logs/cam_ablation_zero_embedding_1h.pid

sleep 3

nohup ./monitor_cam_ablation_previews.sh remove cam_ablation_remove_1h 0 /workspace/imtalker_static/logs/cam_ablation_remove_1h.pid \
  > /workspace/imtalker_static/logs/cam_ablation_remove_preview.log 2>&1 < /dev/null &
echo $! > /workspace/imtalker_static/logs/cam_ablation_remove_preview.pid

nohup ./monitor_cam_ablation_previews.sh zero_embedding cam_ablation_zero_embedding_1h 2 /workspace/imtalker_static/logs/cam_ablation_zero_embedding_1h.pid \
  > /workspace/imtalker_static/logs/cam_ablation_zero_embedding_preview.log 2>&1 < /dev/null &
echo $! > /workspace/imtalker_static/logs/cam_ablation_zero_embedding_preview.pid

echo remove_pid=$(cat /workspace/imtalker_static/logs/cam_ablation_remove_1h.pid)
echo zero_pid=$(cat /workspace/imtalker_static/logs/cam_ablation_zero_embedding_1h.pid)
echo remove_preview_pid=$(cat /workspace/imtalker_static/logs/cam_ablation_remove_preview.pid)
echo zero_preview_pid=$(cat /workspace/imtalker_static/logs/cam_ablation_zero_embedding_preview.pid)
