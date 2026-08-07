#!/usr/bin/env bash
# Run ComfyUI with uv. On first run this creates the virtualenv (if missing)
# and installs all dependencies from requirements.txt, then starts the server.
# Extra CLI args are passed through to main.py, e.g.:
#   ./start_uv.sh --port 8188 --disable-auto-launch
set -euo pipefail
cd "$(dirname "$0")"

uv venv --allow-existing .venv
uv pip install --python .venv/bin/python -r requirements.txt
exec uv run main.py "$@"
