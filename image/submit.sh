#!/usr/bin/env bash
# Submit wrapper installed at /usr/local/bin/submit inside the agent container.
# Validates a candidate submission directory against the baseline record by
# running the comparability judge and the statistical training-run check.
set -euo pipefail
exec /opt/venv/bin/python3 /opt/nanogpt/tools/submit.py "$@"
