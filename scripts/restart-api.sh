#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-javelin-api}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/check}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"

echo "==> Restarting ${SERVICE_NAME}"
sudo systemctl restart "$SERVICE_NAME"

echo "==> Waiting for health check"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if curl --fail --silent "$HEALTH_URL" >/dev/null; then
    echo "Health check passed on attempt ${attempt}"
    curl --silent "$HEALTH_URL"
    echo
    exit 0
  fi

  status="$(systemctl is-active "$SERVICE_NAME" 2>/dev/null || true)"
  echo "Attempt ${attempt}/${MAX_ATTEMPTS}: not ready yet (service: ${status:-unknown})"
  sleep "$SLEEP_SECONDS"
done

echo "ERROR: API did not become healthy"
echo
systemctl status "$SERVICE_NAME" --no-pager -l || true
echo
journalctl -u "$SERVICE_NAME" -n 40 --no-pager || true
exit 1
