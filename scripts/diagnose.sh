#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/mybackend_bh}"
cd "$APP_DIR"

echo "=== Python ==="
python3 --version
venv/bin/python --version

echo
echo "=== Model files ==="
ls -lh neext/final_model_attention.keras neext/scaler.save yolov8n.pt 2>&1 || true

echo
echo "=== App import (lightweight) ==="
venv/bin/python -c "from app import app; print('app import ok')"

echo
echo "=== Service status ==="
systemctl status javelin-api --no-pager -l || true

echo
echo "=== Port 8000 ==="
sudo ss -ltnp | grep ':8000' || echo "Nothing listening on port 8000"

echo
echo "=== Recent service logs ==="
journalctl -u javelin-api -n 40 --no-pager || true

echo
echo "=== Health check ==="
curl -v http://127.0.0.1:8000/check || true
