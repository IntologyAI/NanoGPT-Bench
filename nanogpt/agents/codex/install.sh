#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y ripgrep
$SUDO npm install -g @openai/codex@latest
