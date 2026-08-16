#!/usr/bin/env bash
set -Eeuo pipefail

agent_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$agent_dir" rev-parse --show-toplevel)
task_file="$agent_dir/TASK.md"
report_file="$agent_dir/AGENT_REPORT.md"
codex_bin=${CODEX_BIN:-codex}

for required in GOAL.md STATE.md TASK.md AGENT_REPORT.md; do
  if [[ ! -s "$agent_dir/$required" ]]; then
    echo "Missing or empty .agent/$required" >&2
    exit 2
  fi
done

branch=$(git -C "$repo_root" branch --show-current)
if [[ -z "$branch" || "$branch" == main || "$branch" == master ]]; then
  echo "Refusing to execute on branch '${branch:-DETACHED}'. Use agent-work or an approved development branch." >&2
  exit 3
fi

if ! grep -Eq '^Status:[[:space:]]*READY[[:space:]]*$' "$task_file"; then
  echo "No READY planner task in .agent/TASK.md; nothing executed." >&2
  exit 4
fi

if ! command -v "$codex_bin" >/dev/null 2>&1; then
  echo "Codex executable not found: $codex_bin" >&2
  exit 5
fi

{
  printf '%s\n' \
    'You are the Executor. GPT/ChatGPT is the Planner.' \
    'Execute only the bounded objective in .agent/TASK.md.' \
    'You may inspect/edit code, run tests or experiments, and debug only as required by that objective.' \
    'Do not choose a new high-level direction or start another objective.' \
    'When finished or when a research decision is needed, update .agent/AGENT_REPORT.md.' \
    'The report must end with exactly one of: TASK_COMPLETE, NEED_DECISION, BLOCKED, ERROR.' \
    'Do not work on main/master. Commit completed work and the report, record the relevant SHA, and push the current development branch when a remote is configured.' \
    '' \
    'Repository goal:'
  cat "$agent_dir/GOAL.md"
  printf '\n%s\n' 'Current repository state:'
  cat "$agent_dir/STATE.md"
  printf '\n%s\n' 'Planner-assigned task (the only objective you may execute):'
  cat "$task_file"
} | "$codex_bin" exec --sandbox workspace-write --skip-git-repo-check -C "$repo_root" -

final_status=$(tail -n 1 "$report_file" | tr -d '\r')
if [[ ! "$final_status" =~ ^(TASK_COMPLETE|NEED_DECISION|BLOCKED|ERROR)$ ]]; then
  echo "AGENT_REPORT.md must end with exactly one valid status; found: $final_status" >&2
  exit 6
fi

printf 'Executor finished with status: %s\n' "$final_status"
