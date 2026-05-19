#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -r "${BENCHMARK_CLOCK_HELPER:-/opt/nanogpt-lock/clock.sh}" ]]; then
  # shellcheck source=/dev/null
  source "${BENCHMARK_CLOCK_HELPER:-/opt/nanogpt-lock/clock.sh}"
else
  benchmark_now() { date +%s; }
fi

problem_file=""
rules_file=""
prompt_file=""
resume_file=""

while (($#)); do
  case "$1" in
    --problem-file)
      problem_file="${2:-}"
      shift 2
      ;;
    --problem-file=*)
      problem_file="${1#*=}"
      shift
      ;;
    --rules-file)
      rules_file="${2:-}"
      shift 2
      ;;
    --rules-file=*)
      rules_file="${1#*=}"
      shift
      ;;
    --prompt-file)
      prompt_file="${2:-}"
      shift 2
      ;;
    --prompt-file=*)
      prompt_file="${1#*=}"
      shift
      ;;
    --resume-file)
      resume_file="${2:-}"
      shift 2
      ;;
    --resume-file=*)
      resume_file="${1#*=}"
      shift
      ;;
    *)
      printf 'unknown claude argument: %s\n' "$1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$rules_file" ]]; then
  printf '%s\n' 'missing --rules-file for claude agent' >&2
  exit 1
fi

if [[ -z "$prompt_file" ]]; then
  printf '%s\n' 'missing --prompt-file for claude agent' >&2
  exit 1
fi

if [[ -z "$resume_file" ]]; then
  printf '%s\n' 'missing --resume-file for claude agent' >&2
  exit 1
fi

problem_text=""
if [[ -n "$problem_file" ]]; then
  problem_text="$(<"$problem_file")"
fi
prompt_text="$(<"$prompt_file")"
resume_prompt="$(<"$resume_file")"
prompt="$(<"$rules_file")"$'\n\n'"$prompt_text"
if [[ -n "$problem_text" ]]; then
  prompt="$problem_text"$'\n\n'"$prompt"
fi
session_hours="${BENCHMARK_SESSION_HOURS:-}"
session_seconds=0

if [[ -n "$session_hours" ]]; then
  if [[ ! "$session_hours" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    printf 'invalid BENCHMARK_SESSION_HOURS: %s\n' "$session_hours" >&2
    exit 1
  fi
  if [[ "$(awk -v value="$session_hours" 'BEGIN { print (value > 0) ? 1 : 0 }')" == "1" ]]; then
    session_seconds="$(awk -v value="$session_hours" 'BEGIN { seconds = int(value * 3600); print (seconds < 1) ? 1 : seconds }')"
  fi
fi

if (( session_seconds > 0 )) && [[ -z "${resume_prompt//[[:space:]]/}" ]]; then
  printf 'missing resume prompt in %s\n' "$resume_file" >&2
  exit 1
fi

experiment_dir="${BENCHMARK_EXPERIMENT_DIR:-$BENCHMARK_WORKSPACE/experiments}"

mkdir -p "$BENCHMARK_TRACE_DIR" "$BENCHMARK_LOG_DIR" "$BENCHMARK_SUBMISSION_DIR" "$experiment_dir"
cp -f "$rules_file" "$BENCHMARK_WORKSPACE/RULES.md"

export HOME="$BENCHMARK_TRACE_DIR"

attempt_log="$BENCHMARK_TRACE_DIR/claude-attempt.jsonl"
attempt_text="$BENCHMARK_TRACE_DIR/claude-attempt.txt"
readable_events_path="$BENCHMARK_WORKSPACE/agent_trace.txt"
session_path="$BENCHMARK_TRACE_DIR/claude-session-id"
session_id=""

if [[ -f "$session_path" ]]; then
  session_id="$(<"$session_path")"
fi

run_claude() {
  local prompt_text="$1"
  local current_session_id="$2"

  set +e
  if [[ -n "$current_session_id" ]]; then
    claude \
      --resume "$current_session_id" \
      -p "$prompt_text" \
      --model claude-opus-4-6 \
      --effort max \
      --allowedTools "Bash,Edit,NotebookEdit,Skill,Write" \
      --disallowedTools "WebFetch,WebSearch" \
      --output-format stream-json \
      --verbose \
      --include-partial-messages | tee "$attempt_log" | tee -a "$BENCHMARK_EVENTS_PATH" | python3 -u "$script_dir/render.py" | tee "$attempt_text" | tee -a "$readable_events_path"
  else
    claude \
      -p "$prompt_text" \
      --model claude-opus-4-6 \
      --effort max \
      --allowedTools "Bash,Edit,NotebookEdit,Skill,Write" \
      --disallowedTools "WebFetch,WebSearch" \
      --output-format stream-json \
      --verbose \
      --include-partial-messages | tee "$attempt_log" | tee -a "$BENCHMARK_EVENTS_PATH" | python3 -u "$script_dir/render.py" | tee "$attempt_text" | tee -a "$readable_events_path"
  fi
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

capture_session() {
  sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p' "$attempt_log" | sed -n '1p'
}

if [[ "${BENCHMARK_SKIP_AGENT_INSTALL:-}" != "1" ]]; then
  bash "$script_dir/install.sh"
fi

if [[ -f "$BENCHMARK_WORKSPACE/modal_run.py" ]]; then
  uv pip install --python /opt/venv/bin/python modal
fi

cd "$BENCHMARK_WORKSPACE"

if (( session_seconds == 0 )); then
  : > "$BENCHMARK_EVENTS_PATH"
  : > "$readable_events_path"
  if run_claude "$prompt" ""; then
    exit 0
  else
    exit $?
  fi
fi

touch "$BENCHMARK_EVENTS_PATH"
touch "$readable_events_path"

deadline_epoch=$(( $(benchmark_now) + session_seconds ))
attempt=1
last_status=0

while :; do
  if (( attempt > 1 )) && (( $(benchmark_now) >= deadline_epoch )); then
    break
  fi

  current_prompt="$prompt"
  current_session_id="$session_id"
  if [[ -n "$current_session_id" ]]; then
    current_prompt="$resume_prompt"
  fi

  if run_claude "$current_prompt" "$current_session_id"; then
    last_status=0
  else
    last_status=$?
  fi

  if [[ -z "$session_id" ]]; then
    session_id="$(capture_session)"
    if [[ -z "$session_id" ]]; then
      printf '%s\n' 'claude session resume enabled but no session_id was emitted' >&2
      exit 1
    fi
    printf '%s\n' "$session_id" > "$session_path"
  fi

  attempt=$((attempt + 1))
done

exit "$last_status"
