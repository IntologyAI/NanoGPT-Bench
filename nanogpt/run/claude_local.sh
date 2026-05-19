#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd "$script_dir/.." && pwd)"
agents_dir="$bench_dir/agents"
gpus="${BENCHMARK_GPUS:-all}"

command=(
  python3 "$bench_dir/driver.py"
  --agent "$agents_dir/claude"
  --support-dir "prompts=$bench_dir/prompts"
  --agent-arg=--problem-file=/runner/prompts/problem.txt
  --agent-arg=--rules-file=/runner/prompts/RULES.md
  --agent-arg=--prompt-file=/runner/prompts/local_prompt.md
  --agent-arg=--resume-file=/runner/prompts/resume_prompt.md
  --pass-env ANTHROPIC_API_KEY
  --pass-env OPENAI_API_KEY
  --pass-env CODEX_API_KEY
  --pass-env OPENAI_BASE_URL
  --pass-env AZURE_API_KEY
  --pass-env AZURE_OPENAI_API_KEY
  --pass-env AZURE_API_BASE
  --pass-env AZURE_OPENAI_BASE_URL
  --pass-env AZURE_API_VERSION
  --pass-env AZURE_OPENAI_DEPLOYMENT
  --pass-env BENCHMARK_CODEX_MODEL
  --pass-env BENCHMARK_SESSION_HOURS
)

if [[ -n "$gpus" ]]; then
  command+=( --gpus "$gpus" )
fi

command+=(
  --run-name claude-local
  "$@"
)

"${command[@]}"
