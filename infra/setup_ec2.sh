#!/bin/bash
# infra/setup_ec2.sh  — [Partner B]
#
# Run ONCE on the EC2 instance via SSH after first launch, OR re-run at any
# time to apply the new port additions — all steps are idempotent.
#
# WHAT CHANGED IN THIS VERSION vs the original:
#   1. UFW rules added for 6 new honeypot ports:
#        2222 (SSH alt), 23 (Telnet), 3306 (MySQL),
#        5432 (PostgreSQL), 6379 (Redis), 27017 (MongoDB)
#   2. iptables rules to redirect port 80 → 5001 and port 8080 → 5001
#        Port 80 is one of the internet's most-scanned ports. Redirecting
#        it to the web honeypot (5001) dramatically increases text data volume
#        with zero code change to Flask/gunicorn. Flask never sees port 80
#        directly — iptables NAT rewrites the destination before the packet
#        reaches the socket.
#   3. iptables-persistent installed so these NAT rules survive reboots.
#   4. web-honeypot.service ExecStart now calls gunicorn DIRECTLY at its full
#        path, and WorkingDirectory is set to .../honeypots/ so that the module
#        string "web_honeypot:app" resolves correctly.
#        (This fixes the bug documented in README §4.)
#   5. network-honeypot.service WorkingDirectory unchanged (/home/ubuntu/aml_fintech)
#        since the new multi-port script uses Path(__file__).resolve() to locate
#        data dirs, not the process CWD.
#
# IMPORTANT — before running this script:
#   Go to EC2 Console → Security Groups → Edit Inbound Rules → add:
#     TCP 2222  from 0.0.0.0/0   (SSH alt / honeypot)
#     TCP 23    from 0.0.0.0/0   (Telnet  / honeypot)
#     TCP 80    from 0.0.0.0/0   (HTTP    / redirected to web honeypot)
#     TCP 8080  from 0.0.0.0/0   (HTTP alt / redirected to web honeypot)
#     TCP 3306  from 0.0.0.0/0   (MySQL   / honeypot)
#     TCP 5432  from 0.0.0.0/0   (Postgres/ honeypot)
#     TCP 6379  from 0.0.0.0/0   (Redis   / honeypot)
#     TCP 27017 from 0.0.0.0/0   (MongoDB / honeypot)
#   The Security Group check happens BEFORE packets reach the instance,
#   so UFW and iptables rules have no effect if the SG blocks the port.

set -e

BUCKET_NAME="aml-fintech-honeypot-dataset"

echo "========================================================="
echo "AML FinTech EC2 Setup (idempotent — safe to re-run)"
echo "========================================================="

# ── [1/10] System update ───────────────────────────────────────────────────── #
echo ""
echo "[1/10] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# ── [2/10] System dependencies ────────────────────────────────────────────── #
echo ""
echo "[2/10] Installing Python 3.10, pip, git, ufw, unzip, curl, iptables-persistent..."
sudo apt-get install -y python3.10 python3-pip git ufw unzip curl

# iptables-persistent saves NAT rules to /etc/iptables/rules.v4 so they
# survive reboots. The DEBIAN_FRONTEND=noninteractive flag skips the
# interactive "save current rules?" dialog during apt install.
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y iptables-persistent

# ── [3/10] AWS CLI v2 ─────────────────────────────────────────────────────── #
echo ""
echo "[3/10] Checking / installing AWS CLI v2..."
echo "       (Ubuntu 24.04's apt repos do not carry 'awscli' — installing from AWS directly)"
if ! command -v aws &> /dev/null; then
    cd /tmp
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -o -q awscliv2.zip
    sudo ./aws/install --update
    cd ~
else
    echo "       aws cli already installed — skipping."
fi
aws --version

# ── [4/10] Python packages ────────────────────────────────────────────────── #
echo ""
echo "[4/10] Installing Python packages (--break-system-packages for Ubuntu 24.04 PEP 668)..."
pip3 install --user --break-system-packages \
    flask gunicorn pandas numpy sqlmodel requests

# Verify gunicorn is reachable at the path the service unit will use
GUNICORN_PATH="$HOME/.local/bin/gunicorn"
if [ ! -f "$GUNICORN_PATH" ]; then
    echo "WARNING: gunicorn not found at $GUNICORN_PATH"
    echo "         Run: which gunicorn   to find the correct path"
    echo "         Then update web-honeypot.service ExecStart accordingly."
else
    echo "       gunicorn found at: $GUNICORN_PATH"
fi

# ── [5/10] Directory structure ────────────────────────────────────────────── #
echo ""
echo "[5/10] Creating project directory structure..."
mkdir -p ~/aml_fintech/data/raw/tabular
mkdir -p ~/aml_fintech/data/raw/text
mkdir -p ~/aml_fintech/data/raw/image
mkdir -p ~/aml_fintech/honeypots
mkdir -p ~/aml_fintech/logs
mkdir -p ~/aml_fintech/scripts
mkdir -p ~/aml_fintech/configs

# Ensure the ubuntu user owns everything (prevents the PermissionError
# documented in README §3 that occurs when sudo accidentally creates root-owned files)
sudo chown -R ubuntu:ubuntu ~/aml_fintech

# ── [6/10] S3 sync cron job ───────────────────────────────────────────────── #
echo ""
echo "[6/10] Setting up S3 sync cron job (every 30 minutes)..."
# Use grep -v to remove any existing aml_fintech sync line before re-adding,
# making this step safe to re-run without duplicate cron entries.
( crontab -l 2>/dev/null | grep -v "aml_fintech/data/" \
  ; echo "*/30 * * * * aws s3 sync ~/aml_fintech/data/ s3://${BUCKET_NAME}/data/ --quiet" \
) | crontab -
echo "       Cron job registered. Verify with: crontab -l"

# ── [7/10] iptables NAT rules (port 80 + 8080 → 5001) ────────────────────── #
echo ""
echo "[7/10] Configuring iptables PREROUTING redirects..."
echo "       Port 80  → 5001 (web honeypot)"
echo "       Port 8080 → 5001 (web honeypot)"

# -C checks if a rule exists before adding it.
# If the check exits non-zero (rule doesn't exist), we add it with -A.
# This makes the step idempotent — re-running won't duplicate the rule.

if ! sudo iptables -t nat -C PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 5001 2>/dev/null; then
    sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 5001
    echo "       Added: port 80 → 5001"
else
    echo "       Already present: port 80 → 5001 (skipping)"
fi

if ! sudo iptables -t nat -C PREROUTING -p tcp --dport 8080 -j REDIRECT --to-port 5001 2>/dev/null; then
    sudo iptables -t nat -A PREROUTING -p tcp --dport 8080 -j REDIRECT --to-port 5001
    echo "       Added: port 8080 → 5001"
else
    echo "       Already present: port 8080 → 5001 (skipping)"
fi

# Persist iptables rules so they survive reboots
sudo netfilter-persistent save
echo "       iptables rules saved (survive reboot via iptables-persistent)."

# ── [8/10] systemd service files ──────────────────────────────────────────── #
echo ""
echo "[8/10] Writing systemd service files..."

# ── network-honeypot.service ──────────────────────────────────────────────── #
sudo tee /etc/systemd/system/network-honeypot.service > /dev/null <<'SERVICE_EOF'
# infra/network-honeypot.service
[Unit]
Description=AML FinTech Network Honeypot (Multi-Port)
After=network.target

[Service]
Type=simple
User=ubuntu
# WorkingDirectory is the project root. The script itself resolves all
# data/log paths from Path(__file__).resolve() so this is not critical,
# but it's the conventional place for a Python project.
WorkingDirectory=/home/ubuntu/aml_fintech
ExecStart=/usr/bin/python3 honeypots/network_honeypot.py
Restart=always
RestartSec=10
StandardOutput=append:/home/ubuntu/aml_fintech/logs/network_honeypot.log
StandardError=append:/home/ubuntu/aml_fintech/logs/network_honeypot.log

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# ── web-honeypot.service ───────────────────────────────────────────────────── #
# WorkingDirectory MUST be the honeypots/ folder, not the project root.
# Gunicorn resolves "web_honeypot:app" as a Python module name, so it must
# be importable from WorkingDirectory. With WorkingDirectory=honeypots/,
# Python can find web_honeypot.py directly. The script itself uses
# Path(__file__).resolve().parent.parent to locate data dirs, so setting
# WorkingDirectory=honeypots/ does NOT break data file paths.
sudo tee /etc/systemd/system/web-honeypot.service > /dev/null <<'SERVICE_EOF'
# infra/web-honeypot.service
[Unit]
Description=AML FinTech Web Honeypot
After=network.target

[Service]
Type=simple
User=ubuntu
# IMPORTANT: WorkingDirectory must be honeypots/ (not the project root)
# so that gunicorn can resolve "web_honeypot:app" as a Python module.
WorkingDirectory=/home/ubuntu/aml_fintech/honeypots
# Use the full path to gunicorn (installed under ~/.local by pip --user).
# If gunicorn was installed elsewhere, adjust this path.
ExecStart=/home/ubuntu/.local/bin/gunicorn -w 4 -b 0.0.0.0:5001 web_honeypot:app
Restart=always
RestartSec=10
StandardOutput=append:/home/ubuntu/aml_fintech/logs/web_honeypot.log
StandardError=append:/home/ubuntu/aml_fintech/logs/web_honeypot.log

[Install]
WantedBy=multi-user.target
SERVICE_EOF

echo "       Service files written."

# ── [9/10] Enable (but NOT start) services ────────────────────────────────── #
echo ""
echo "[9/10] Reloading systemd and enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable network-honeypot web-honeypot
# Not starting here — honeypot scripts are deployed via deploy_honeypots.sh.
# Starting before the scripts exist produces failed-unit noise.
echo "       Services enabled. Run deploy_honeypots.sh to deploy scripts and start them."

# ── [10/10] UFW firewall ───────────────────────────────────────────────────── #
echo ""
echo "[10/10] Configuring UFW firewall rules..."

# Original ports
sudo ufw allow 22/tcp     # SSH — restricted to your IP in EC2 Security Group,
                           # but UFW also needs it or you'll lock yourself out.
sudo ufw allow 9999/tcp   # Original custom honeypot port
sudo ufw allow 5001/tcp   # Web honeypot (Flask/gunicorn)

# New ports — these ALSO need to be open in the EC2 Security Group
sudo ufw allow 2222/tcp   # SSH alternative (attracts SSH brute-force bots)
sudo ufw allow 23/tcp     # Telnet (attracts Mirai/IoT botnets)
sudo ufw allow 80/tcp     # HTTP (iptables redirects this to 5001)
sudo ufw allow 8080/tcp   # HTTP alt (iptables redirects this to 5001)
sudo ufw allow 3306/tcp   # MySQL (attracts DB exploit scanners)
sudo ufw allow 5432/tcp   # PostgreSQL (attracts DB scanners)
sudo ufw allow 6379/tcp   # Redis (attracts Redis exploit bots)
sudo ufw allow 27017/tcp  # MongoDB (attracts Mongo exploit bots)

sudo ufw --force enable
echo "       UFW configured. Current rules:"
sudo ufw status numbered

echo ""
echo "========================================================="
echo "✅ EC2 setup complete."
echo ""
echo "NEXT STEPS:"
echo "  1. Verify Security Group inbound rules in AWS Console:"
echo "     Open these ports to 0.0.0.0/0 — 2222, 23, 80, 8080, 3306,"
echo "     5432, 6379, 27017  (9999 and 5001 already open from before)"
echo ""
echo "  2. From your laptop, run infra/deploy_honeypots.sh"
echo "     to copy scripts to EC2 and start both honeypots."
echo ""
echo "  3. Verify iptables redirect is working:"
echo "     curl http://16.171.191.100/health   # should return {\"status\":\"ok\"}"
echo "     curl http://16.171.191.100:80/health # same result"
echo ""
echo "  4. Monitor traffic after 24h:"
echo "     bash infra/honeypot_status.sh"
echo "========================================================="