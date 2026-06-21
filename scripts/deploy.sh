#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/mybackend_bh}"
SERVICE_NAME="${SERVICE_NAME:-javelin-api}"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"
BRANCH="${BRANCH:-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=resolve-python.sh
source "$SCRIPT_DIR/resolve-python.sh"

echo "==> Deploying from $APP_DIR"

cd "$APP_DIR"

if [ ! -d .git ]; then
  echo "ERROR: $APP_DIR is not a git repository"
  exit 1
fi

echo "==> Pulling latest code"
git fetch origin "$BRANCH"
git reset --hard "origin/$BRANCH"

chmod +x scripts/deploy.sh scripts/resolve-python.sh

ensure_python

if [ ! -d "$VENV_DIR" ] || ! "$VENV_DIR/bin/python" -c "import sys; exit(0 if sys.version_info[:2] in ((3,12),(3,11)) else 1)" 2>/dev/null; then
  echo "==> Creating virtual environment with $PYTHON_BIN"
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "==> Installing dependencies ($(python --version))"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "==> Restarting service"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "==> Waiting for service"
sleep 3
sudo systemctl is-active --quiet "$SERVICE_NAME"

echo "==> Local health check"
curl --fail --retry 10 --retry-delay 3 http://127.0.0.1:8000/check

echo "Deploy completed successfully"
