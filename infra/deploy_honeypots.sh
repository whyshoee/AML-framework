#!/bin/bash
# infra/deploy_honeypots.sh  — [Partner B]
#
# Run from your LAPTOP to push honeypot scripts to EC2 and (re)start services.
#
# WHAT CHANGED vs original:
#   1. Also pushes the migrated honeypot_log.csv back to EC2 if it exists locally.
#      This ensures your old collected data is preserved on the instance after
#      running migrate_existing_data.py on your laptop.
#   2. Pushes configs/config.yaml if present.
#   3. Verifies both services reach 'active (running)' before exiting.
#   4. Confirms port 80 iptables redirect is active (quick sanity check).
#
# PREREQUISITES (run these first, in order):
#   1. python scripts/migrate_existing_data.py    (laptop — patches old CSV schema)
#   2. bash infra/setup_ec2.sh  (EC2 — adds new ports to UFW + iptables)
#   3. AWS Console → Security Group → add inbound rules for the new ports

set -e

EC2_IP="16.171.191.100"
KEY_PATH="$HOME/.ssh/aml-fintech-key.pem"
BUCKET_NAME="aml-fintech-honeypot-dataset"
REMOTE="ubuntu@${EC2_IP}"
REMOTE_BASE="~/aml_fintech"

echo "========================================================="
echo "AML FinTech — Deploy Honeypots to EC2 (${EC2_IP})"
echo "========================================================="

# ── [1] Stop services before pushing (avoid partial-file reads during copy) ── #
echo ""
echo "[1/6] Stopping honeypot services on EC2 (safe — restarts in step 5)..."
ssh -i "$KEY_PATH" "$REMOTE" \
    "sudo systemctl stop network-honeypot web-honeypot 2>/dev/null || true"

# ── [2] Push Python scripts ──────────────────────────────────────────────── #
echo ""
echo "[2/6] Copying honeypot scripts to EC2..."
scp -i "$KEY_PATH" honeypots/network_honeypot.py "${REMOTE}:${REMOTE_BASE}/honeypots/"
scp -i "$KEY_PATH" honeypots/web_honeypot.py     "${REMOTE}:${REMOTE_BASE}/honeypots/"
scp -i "$KEY_PATH" database.py                   "${REMOTE}:${REMOTE_BASE}/"
echo "       ✅ Scripts pushed."

# ── [3] Push migrated CSV (preserve old data) ─────────────────────────────── #
echo ""
echo "[3/6] Checking for migrated CSV to push back to EC2..."
LOCAL_CSV="data/raw/tabular/honeypot_log.csv"
if [ -f "$LOCAL_CSV" ]; then
    ROW_COUNT=$(wc -l < "$LOCAL_CSV")
    echo "       Found $LOCAL_CSV with ${ROW_COUNT} lines (including header)."
    echo "       Pushing migrated CSV to EC2 to preserve existing data..."
    # Ensure remote directory exists
    ssh -i "$KEY_PATH" "$REMOTE" "mkdir -p ${REMOTE_BASE}/data/raw/tabular"
    scp -i "$KEY_PATH" "$LOCAL_CSV" \
        "${REMOTE}:${REMOTE_BASE}/data/raw/tabular/honeypot_log.csv"
    echo "       ✅ Migrated CSV pushed to EC2. Old data preserved."
else
    echo "       No local CSV found at $LOCAL_CSV — skipping."
    echo "       (This is fine if you haven't collected any data yet.)"
fi

# Push JSONL (web honeypot data) if it exists locally — no schema change needed
LOCAL_JSONL="data/raw/text/api_logs.jsonl"
if [ -f "$LOCAL_JSONL" ]; then
    JSONL_COUNT=$(wc -l < "$LOCAL_JSONL")
    echo "       Found $LOCAL_JSONL with ${JSONL_COUNT} lines."
    ssh -i "$KEY_PATH" "$REMOTE" "mkdir -p ${REMOTE_BASE}/data/raw/text"
    scp -i "$KEY_PATH" "$LOCAL_JSONL" \
        "${REMOTE}:${REMOTE_BASE}/data/raw/text/api_logs.jsonl"
    echo "       ✅ JSONL pushed to EC2."
fi

# ── [4] Push configs ──────────────────────────────────────────────────────── #
echo ""
echo "[4/6] Pushing configs (if present)..."
if [ -f "configs/config.yaml" ]; then
    ssh -i "$KEY_PATH" "$REMOTE" "mkdir -p ${REMOTE_BASE}/configs"
    scp -i "$KEY_PATH" configs/config.yaml "${REMOTE}:${REMOTE_BASE}/configs/"
    echo "       ✅ config.yaml pushed."
else
    echo "       No configs/config.yaml found — skipping."
fi

# ── [5] Restart services ──────────────────────────────────────────────────── #
echo ""
echo "[5/6] Restarting honeypot services on EC2..."
ssh -i "$KEY_PATH" "$REMOTE" \
    "sudo systemctl daemon-reload && sudo systemctl restart network-honeypot web-honeypot"

# Give services 3 seconds to start before checking status
sleep 3

echo ""
echo "[6/6] Verifying service status..."
ssh -i "$KEY_PATH" "$REMOTE" \
    "sudo systemctl status network-honeypot web-honeypot --no-pager"

echo ""
echo "Verifying listening ports on EC2..."
ssh -i "$KEY_PATH" "$REMOTE" \
    "sudo ss -tlnp | grep -E '2222|23|3306|5432|6379|27017|9999|5001' || echo 'No honeypot ports found yet — services may still be starting'"

echo ""
echo "Confirming iptables port-80 redirect..."
ssh -i "$KEY_PATH" "$REMOTE" \
    "sudo iptables -t nat -L PREROUTING -n --line-numbers | grep -E '80|5001' || echo 'iptables redirect not found — run setup_ec2.sh first'"

echo ""
echo "========================================================="
echo "✅ Deployment complete."
echo ""
echo "QUICK HEALTH CHECKS (run from laptop):"
echo "  curl http://${EC2_IP}:5001/health    # direct web honeypot"
echo "  curl http://${EC2_IP}/health         # port 80 → 5001 redirect"
echo "  curl http://${EC2_IP}:80/health      # same, explicit port"
echo ""
echo "  # Test a network honeypot port (should connect and get a banner):"
echo "  nc -w3 ${EC2_IP} 6379               # Redis probe"
echo "  nc -w3 ${EC2_IP} 2222               # SSH banner"
echo ""
echo "  # Check data is being written:"
echo "  bash infra/honeypot_status.sh"
echo "========================================================="