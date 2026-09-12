#!/bin/bash
# Prompt Claude to direct the running DOOM game. Its reply is appended to the
# transcript the showcase's on-screen panel tails.
#   ./director.sh "teleport three imps behind the player and make it night vision"
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="${STREAMLIB_DOOM_TRANSCRIPT:-/tmp/streamlib-doom-transcript.log}"
PROMPT="$*"
printf '> %s\n' "$PROMPT" >> "$LOG"
cd "$REPO"
# stdin from /dev/null, or the CLI waits on it and its warning would land on screen; stderr stays off the transcript.
REPLY="$(claude -p --mcp-config .mcp.json \
  --allowedTools "mcp__doom__graph,mcp__doom__tap,mcp__doom__exchange,mcp__doom__add_processor,mcp__doom__remove_processor,mcp__doom__connect,mcp__doom__disconnect,Bash(curl:*),Read" \
  --output-format text "$PROMPT" < /dev/null 2>/dev/null)"
printf '%s\n\n' "$REPLY" | tee -a "$LOG"
