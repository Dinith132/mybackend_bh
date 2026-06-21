#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/mybackend_bh}"
SITE_NAME="${SITE_NAME:-javelin-api}"

echo "==> Configuring Nginx reverse proxy"
sudo cp "$APP_DIR/deploy/nginx-javelin-api.conf" "/etc/nginx/sites-available/${SITE_NAME}"
sudo ln -sf "/etc/nginx/sites-available/${SITE_NAME}" /etc/nginx/sites-enabled/javelin-api
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

echo "==> Nginx is proxying port 80 -> 127.0.0.1:8000"
echo "    Open TCP port 80 in your EC2 security group for external access."
