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
      printf 'unknown codex argument: %s\n' "$1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$rules_file" ]]; then
  printf '%s\n' 'missing --rules-file for codex agent' >&2
  exit 1
fi

if [[ -z "$prompt_file" ]]; then
  printf '%s\n' 'missing --prompt-file for codex agent' >&2
  exit 1
fi

if [[ -z "$resume_file" ]]; then
  printf '%s\n' 'missing --resume-file for codex agent' >&2
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
export CODEX_HOME="/var/lib/codex"

mkdir -p "$CODEX_HOME"
chmod 700 "$CODEX_HOME"

attempt_log="$BENCHMARK_TRACE_DIR/codex-attempt.jsonl"
session_path="$BENCHMARK_TRACE_DIR/codex-session-id"
session_id=""
config_path="$CODEX_HOME/config.toml"
azure_key="${AZURE_API_KEY:-${AZURE_OPENAI_API_KEY:-}}"
azure_base="${AZURE_OPENAI_BASE_URL:-${AZURE_API_BASE:-${OPENAI_BASE_URL:-}}}"
azure_version="${AZURE_API_VERSION:-2025-04-01-preview}"
openai_model="${BENCHMARK_CODEX_MODEL:-gpt-5.4}"
azure_deployment="${AZURE_OPENAI_DEPLOYMENT:-}"
codex_reasoning="${BENCHMARK_CODEX_REASONING_EFFORT:-xhigh}"

if [[ -f "$session_path" ]]; then
  session_id="$(<"$session_path")"
fi

configure_codex() {
  local azure_root=""

  if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${CODEX_API_KEY:-}" ]] && [[ -n "$azure_key" ]]; then
    export AZURE_API_KEY="$azure_key"
    export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-$azure_key}"
    if [[ -z "$azure_base" ]]; then
      printf '%s\n' 'AZURE_API_BASE or AZURE_OPENAI_BASE_URL must be set when using AZURE_API_KEY with Codex' >&2
      exit 1
    fi
    azure_root="${azure_base%/}"
    if [[ -z "$azure_deployment" ]]; then
      printf '%s\n' 'AZURE_OPENAI_DEPLOYMENT must be set when using Azure with Codex' >&2
      exit 1
    fi
    if [[ "$azure_root" != */openai ]]; then
      azure_root="$azure_root/openai"
    fi
    : > "$config_path"
    printf 'model = "%s"\n' "$azure_deployment" >> "$config_path"
    printf 'model_reasoning_effort = "%s"\n' "$codex_reasoning" >> "$config_path"
    printf 'model_provider = "azure"\n' >> "$config_path"
    printf '[model_providers.azure]\n' >> "$config_path"
    printf 'name = "Azure"\n' >> "$config_path"
    printf 'base_url = "%s"\n' "$azure_root" >> "$config_path"
    printf 'env_key = "AZURE_API_KEY"\n' >> "$config_path"
    printf 'query_params = { api-version = "%s" }\n' "$azure_version" >> "$config_path"
    printf 'wire_api = "responses"\n' >> "$config_path"
    unset OPENAI_API_KEY CODEX_API_KEY
  elif [[ -n "${OPENAI_API_KEY:-}" ]] || [[ -n "${CODEX_API_KEY:-}" ]]; then
    : > "$config_path"
    if [[ -n "$openai_model" ]]; then
      printf 'model = "%s"\n' "$openai_model" >> "$config_path"
    fi
    printf 'model_reasoning_effort = "%s"\n' "$codex_reasoning" >> "$config_path"
    local api_key="${OPENAI_API_KEY:-${CODEX_API_KEY:-}}"
    if [[ -n "$api_key" ]]; then
      printf '%s' "$api_key" | codex login --with-api-key
    fi
  fi
}

run_codex() {
  local prompt_text="$1"
  local current_session_id="$2"

  set +e
  if [[ -n "$current_session_id" ]]; then
    codex exec resume \
      --dangerously-bypass-approvals-and-sandbox \
      -c 'web_search="disabled"' \
      --json \
      -o "$BENCHMARK_FINAL_PATH" \
      "$current_session_id" \
      "$prompt_text" < /dev/null | tee "$attempt_log" | tee -a "$BENCHMARK_EVENTS_PATH"
  else
    codex exec \
      --dangerously-bypass-approvals-and-sandbox \
      -c 'web_search="disabled"' \
      --json \
      -o "$BENCHMARK_FINAL_PATH" \
      "$prompt_text" < /dev/null | tee "$attempt_log" | tee -a "$BENCHMARK_EVENTS_PATH"
  fi
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

capture_session() {
  sed -n 's/.*"thread_id":"\([^"]*\)".*/\1/p' "$attempt_log" | sed -n '1p'
}

if [[ "${BENCHMARK_SKIP_AGENT_INSTALL:-}" != "1" ]]; then
  bash "$script_dir/install.sh"
fi

configure_codex

if [[ -f "$BENCHMARK_WORKSPACE/modal_run.py" ]]; then
  uv pip install --python /opt/venv/bin/python modal
fi

cd "$BENCHMARK_WORKSPACE"

if (( session_seconds == 0 )); then
  : > "$BENCHMARK_EVENTS_PATH"
  if run_codex "$prompt" ""; then
    exit 0
  else
    exit $?
  fi
fi

touch "$BENCHMARK_EVENTS_PATH"

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

  if run_codex "$current_prompt" "$current_session_id"; then
    last_status=0
  else
    last_status=$?
  fi

  if [[ -z "$session_id" ]]; then
    session_id="$(capture_session)"
    if [[ -z "$session_id" ]]; then
      printf '%s\n' 'codex session resume enabled but no thread_id was emitted' >&2
      exit 1
    fi
    printf '%s\n' "$session_id" > "$session_path"
  fi

  attempt=$((attempt + 1))
done

exit "$last_status"
