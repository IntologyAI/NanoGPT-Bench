#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then SUDO=sudo; else SUDO=; fi
$SUDO npm install -g @anthropic-ai/claude-code@latest @openai/codex@latest
