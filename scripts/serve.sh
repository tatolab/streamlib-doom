#!/bin/bash
# Keeps the Doom node up: `streamlib run` in this repo, restarted if it ever exits.
# Usage: scripts/serve.sh [path-to-venv-bin]   (defaults to .venv/bin beside this repo)
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${1:-$REPO/.venv/bin}"
cd "$REPO"
while true; do
  "$BIN/streamlib" run --port 9200
  echo "$(date '+%F %T') node exited, restarting in 3 s"
  sleep 3
done
