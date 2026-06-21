#!/usr/bin/env bash
set -euo pipefail

# One-time EC2 setup script.
# Run on a fresh Ubuntu instance:
#   bash scripts/server-bootstrap.sh
#
# Optional:
#   PYTHON_VERSION=3.12|3.11

APP_DIR="${APP_DIR:-/home/ubuntu/mybackend_bh}"
REPO_URL="${REPO_URL:-https://github.com/Dinith132/mybackend_bh.git}"
SERVICE_NAME="${SERVICE_NAME:-javelin-api}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=resolve-python.sh
source "$SCRIPT_DIR/resolve-python.sh"

echo "==> Installing system packages"
sudo apt update
sudo apt install -y \
  git \
  curl \
  nginx \
  libgl1 \
  libsm6 \
  libxext6 \
  libxrender-dev

if [ ! -d "$APP_DIR/.git" ]; then
  echo "==> Cloning repository"
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
chmod +x scripts/deploy.sh scripts/resolve-python.sh

ensure_python

echo "==> Creating virtual environment"
rm -rf venv
"$PYTHON_BIN" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "==> Installing systemd service"
sudo cp deploy/javelin-api.service "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "==> Verifying local health check"
sleep 3
curl --fail http://127.0.0.1:8000/check

echo
echo "Bootstrap complete."
echo "Python: $(venv/bin/python --version)"
echo "Next steps:"
echo "  1. Add GitHub secrets: EC2_HOST, EC2_USER, EC2_SSH_KEY"
echo "  2. Optional: EC2_APP_DIR, EC2_HEALTH_URL"
echo "  3. Push to main to trigger GitHub Actions deploy"
