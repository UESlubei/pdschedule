#!/usr/bin/env bash
set -euo pipefail
APB_PROJECT=/home/sdn/wjh/Ondeviceinfer/source/adaptive_prefill_budget
APB_RUNTIME=/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320_opt2
APB_CODEX=/home/sdn/.nvm/versions/node/v20.20.2/bin/codex
mkdir -p "$APB_RUNTIME/logs"
exec 9>"$APB_RUNTIME/codex-worker.lock"
flock -n 9 || { echo 'An optimization worker is already running.'; exit 1; }
cd "$APB_PROJECT"
export PATH=/home/sdn/.nvm/versions/node/v20.20.2/bin:"$PATH"
export APB_DATA="$APB_RUNTIME"
export PYTHONPATH="$APB_PROJECT"
unset CODEX_THREAD_ID
printf 'running pid=%s started=%s\n' "$$" "$(date -Is)" > "$APB_RUNTIME/worker.status"
trap 'APB_EXIT_CODE=$?; printf "exited code=%s time=%s\n" "$APB_EXIT_CODE" "$(date -Is)" > "$APB_RUNTIME/worker.status"' EXIT
"$APB_CODEX" -c 'approval_policy="never"' exec --sandbox danger-full-access \
  --skip-git-repo-check --cd "$APB_PROJECT" --json \
  --output-last-message "$APB_RUNTIME/worker-final.md" - \
  < "$APB_PROJECT/docs/SERVER_CONTINUATION.md" \
  > "$APB_RUNTIME/logs/codex-worker.jsonl" \
  2> "$APB_RUNTIME/logs/codex-worker.stderr.log"
