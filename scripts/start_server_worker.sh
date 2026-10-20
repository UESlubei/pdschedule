#!/usr/bin/env bash
set -euo pipefail
APB_SESSION=apb-optimization
APB_WORKER=/home/sdn/wjh/Ondeviceinfer/source/adaptive_prefill_budget/scripts/server_worker.sh
if tmux has-session -t "$APB_SESSION" 2>/dev/null; then
  echo "Already running: tmux attach -t $APB_SESSION"
  exit 0
fi
tmux new-session -d -s "$APB_SESSION" "bash $APB_WORKER"
echo "Started on server11: tmux session $APB_SESSION"
echo 'Status: cat /mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320_opt2/worker.status'
echo 'Progress: tail -f /mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320_opt2/logs/codex-worker.jsonl'
