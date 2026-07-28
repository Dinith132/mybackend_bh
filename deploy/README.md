# Javelin Pro Backend Deployment Notes

Quick reference for the EC2 backend deployment.

## Current EC2 service

The backend runs as a `systemd` service:

```bash
javelin-api.service
```

Service file in this repo:

```bash
deploy/javelin-api.service
```

Expected EC2 app path:

```bash
/home/ubuntu/mybackend_bh
```

Expected local API port:

```bash
127.0.0.1:8000
```

Current target EC2 host for CI/CD:

```bash
ec2-13-63-175-128.eu-north-1.compute.amazonaws.com
```

The service runs:

```bash
/home/ubuntu/mybackend_bh/venv/bin/gunicorn -w 1 -b 127.0.0.1:8000 app:app --timeout 300 --graceful-timeout 300 --capture-output --log-level info
```

## Check the backend service

Find the service if you forget the name:

```bash
systemctl list-unit-files --type=service | grep -i api
systemctl list-unit-files --type=service | grep -i javelin
```

Check status:

```bash
sudo systemctl status javelin-api.service
```

See live logs:

```bash
sudo journalctl -u javelin-api.service -f
```

Restart after a code update:

```bash
sudo systemctl restart javelin-api.service
```

Check whether it starts automatically after reboot:

```bash
sudo systemctl is-enabled javelin-api.service
```

## Deploy/update flow

On the EC2 instance:

```bash
cd /home/ubuntu/mybackend_bh
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart javelin-api.service
sudo systemctl status javelin-api.service
```

If using the repo script:

```bash
cd /home/ubuntu/mybackend_bh
./scripts/restart-api.sh
```

## Health/API checks

Local check from EC2:

```bash
curl http://127.0.0.1:8000/check
```

If `/check` is not available, try:

```bash
curl http://127.0.0.1:8000
```

## Nginx

This repo also contains an Nginx config:

```bash
deploy/nginx-javelin-api.conf
```

It proxies public HTTP traffic on port `80` to:

```bash
http://127.0.0.1:8000
```

Check Nginx:

```bash
sudo systemctl status nginx
sudo nginx -t
```

Restart Nginx:

```bash
sudo systemctl restart nginx
```

## GitHub Actions CI/CD

This repo has a GitHub Actions workflow:

```bash
.github/workflows/deploy-ec2.yml
```

It deploys automatically when code is pushed to:

```bash
main
```

It can also be run manually from GitHub:

```bash
Actions -> Deploy to EC2 -> Run workflow
```

### Required GitHub secrets

Add these in GitHub:

```bash
Repository -> Settings -> Secrets and variables -> Actions -> New repository secret
```

Required secrets:

```bash
EC2_HOST=ec2-13-63-175-128.eu-north-1.compute.amazonaws.com
EC2_USER=ubuntu
EC2_SSH_KEY=<contents of the private .pem key>
```

Recommended optional secrets:

```bash
EC2_APP_DIR=/home/ubuntu/mybackend_bh
EC2_SERVICE_NAME=javelin-api
EC2_HEALTH_URL=http://ec2-13-63-175-128.eu-north-1.compute.amazonaws.com
```

Do not include quotation marks around the secret values.

For `EC2_SSH_KEY`, paste the full private key, including:

```bash
-----BEGIN ... PRIVATE KEY-----
...
-----END ... PRIVATE KEY-----
```

### EC2 one-time setup checklist

The EC2 instance must already have the repo cloned:

```bash
cd /home/ubuntu
git clone <repo-url> mybackend_bh
```

The Git remote on EC2 must be able to pull from GitHub:

```bash
cd /home/ubuntu/mybackend_bh
git fetch origin main
```

The `ubuntu` user must be allowed to run these deploy commands with `sudo`:

```bash
sudo cp deploy/javelin-api.service /etc/systemd/system/javelin-api.service
sudo systemctl daemon-reload
sudo systemctl enable javelin-api
sudo systemctl restart javelin-api
```

Check service status:

```bash
sudo systemctl status javelin-api
```

Check logs:

```bash
sudo journalctl -u javelin-api -f
```

### What the workflow does

On every push to `main`, GitHub Actions SSHes into EC2 and runs:

```bash
cd /home/ubuntu/mybackend_bh
bash scripts/deploy.sh
```

The deploy script:

```bash
git fetch origin main
git reset --hard origin/main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart javelin-api
```

## Ngrok setup

The backend listens on `127.0.0.1:8000`, so ngrok should expose port `8000`.

If `ngrok-v3-stable-linux-amd64.tgz` is already on the EC2 instance:

```bash
cd ~
tar -xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/ngrok
ngrok version
```

Add the ngrok auth token:

```bash
ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
```

Start a temporary tunnel:

```bash
ngrok http 8000
```

Ngrok will show a forwarding URL like:

```bash
https://example.ngrok-free.app -> http://localhost:8000
```

Use that `https://...ngrok-free.app` URL as the public API URL for testing.

Test it:

```bash
curl https://example.ngrok-free.app/check
```

Note: free ngrok URLs usually change when ngrok restarts. If the frontend needs a stable URL, reserve a static ngrok domain and run ngrok with that domain.

## Optional: run ngrok as a systemd service

Create:

```bash
sudo nano /etc/systemd/system/ngrok-javelin.service
```

Example service:

```ini
[Unit]
Description=Ngrok tunnel for Javelin API
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/ngrok http 8000
Restart=always
RestartSec=5
User=ubuntu

[Install]
WantedBy=multi-user.target
```

Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ngrok-javelin.service
sudo systemctl start ngrok-javelin.service
sudo systemctl status ngrok-javelin.service
```

Logs:

```bash
sudo journalctl -u ngrok-javelin.service -f
```
