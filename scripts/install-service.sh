#!/bin/bash
# Runs the node as a user-level systemd service that outlives the terminal.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
systemctl --user stop streamlib-doom 2>/dev/null; systemctl --user reset-failed streamlib-doom 2>/dev/null
ENV_ARGS=(--setenv=DISPLAY="${DISPLAY:-:0}")
for name in STREAMLIB_WHIP_URL STREAMLIB_WHEP_URL STREAMLIB_WHIP_BEARER_TOKEN STREAMLIB_DOOM_WAD; do
  [ -n "${!name}" ] && ENV_ARGS+=(--setenv="$name=${!name}")
done
systemd-run --user --unit=streamlib-doom --collect --property=Restart=always --property=RestartSec=3 \
  "${ENV_ARGS[@]}" "$REPO/scripts/serve.sh" "$REPO/.venv/bin"
echo "systemctl --user status streamlib-doom   # stop | restart | status"
echo "loginctl enable-linger                   # keep it running after you log out"
