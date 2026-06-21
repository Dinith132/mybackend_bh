#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/mybackend_bh}"
SERVICE_NAME="${SERVICE_NAME:-javelin-api}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"
BRANCH="${BRANCH:-main}"

echo "==> Deploying from $APP_DIR"

cd "$APP_DIR"

if [ ! -d .git ]; then
  echo "ERROR: $APP_DIR is not a git repository"
  exit 1
fi

echo "==> Pulling latest code"
git fetch origin "$BRANCH"
git reset --hard "origin/$BRANCH"

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "==> Installing dependencies"
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
