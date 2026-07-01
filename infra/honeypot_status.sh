#!/bin/bash
# infra/honeypot_status.sh  — [Partner B]
#
# Quick remote status check from your LAPTOP.
# Shows service health, listening ports, recent log lines, and live row counts.
#
# WHAT CHANGED vs original:
#   1. Now checks all 9 honeypot ports (was: only implicit via service status).
#   2. Shows per-service_type row counts from the CSV if the column exists.
#   3. Checks iptables redirect for port 80 is active.
#   4. Validates gunicorn process is running (not just the service unit).

set -e

EC2_IP="16.171.191.100"
KEY_PATH="$HOME/.ssh/aml-fintech-key.pem"

echo "========================================================="
echo "AML FinTech — Honeypot Status  (${EC2_IP})"
echo "========================================================="

echo ""
echo "── Service status ───────────────────────────────────────"
ssh -i "$KEY_PATH" ubuntu@$EC2_IP \
    "sudo systemctl status network-honeypot web-honeypot --no-pager"

echo ""
echo "── Listening ports ──────────────────────────────────────"
ssh -i "$KEY_PATH" ubuntu@$EC2_IP \
    "sudo ss -tlnp | grep -E '2222|:23 |3306|5432|6379|27017|9999|5001' \
     || echo '  ⚠️  No honeypot ports found — check if services are running'"

echo ""
echo "── iptables port-80 redirect ────────────────────────────"
ssh -i "$KEY_PATH" ubuntu@$EC2_IP \
    "sudo iptables -t nat -L PREROUTING -n | grep -E '(REDIRECT|80)' \
     || echo '  ⚠️  No port-80 redirect rule found — run setup_ec2.sh'"

echo ""
echo "── Network honeypot log (last 20 lines) ─────────────────"
ssh -i "$KEY_PATH" ubuntu@$EC2_IP \
    "tail -20 ~/aml_fintech/logs/network_honeypot.log 2>/dev/null \
     || echo '  No log file yet'"

echo ""
echo "── Web honeypot log (last 20 lines) ─────────────────────"
ssh -i "$KEY_PATH" ubuntu@$EC2_IP \
    "tail -20 ~/aml_fintech/logs/web_honeypot.log 2>/dev/null \
     || echo '  No log file yet'"

echo ""
echo "── Live row counts on EC2 ───────────────────────────────"
ssh -i "$KEY_PATH" ubuntu@$EC2_IP bash << 'REMOTE_EOF'
CSV="~/aml_fintech/data/raw/tabular/honeypot_log.csv"
JSONL="~/aml_fintech/data/raw/text/api_logs.jsonl"

# Expand ~ properly on the remote side
CSV="${HOME}/aml_fintech/data/raw/tabular/honeypot_log.csv"
JSONL="${HOME}/aml_fintech/data/raw/text/api_logs.jsonl"

if [ -f "$CSV" ]; then
    ROWS=$(( $(wc -l < "$CSV") - 1 ))
    echo "  Tabular CSV:  ${ROWS} rows"
    # Per-service_type breakdown if new schema
    python3 - "$CSV" <<'PYEOF'
import csv, sys
from collections import Counter
path = sys.argv[1]
try:
    with open(path) as f:
        reader = csv.DictReader(f)
        if 'service_type' in (reader.fieldnames or []):
            c = Counter(row['service_type'] for row in reader)
            for stype, n in sorted(c.items(), key=lambda x: -x[1]):
                print(f"    {stype:<12} {n:>8,} rows")
        else:
            print("    (old schema — service_type column not present yet)")
except Exception as e:
    print(f"    (could not parse: {e})")
PYEOF
else
    echo "  Tabular CSV:  not found"
fi

if [ -f "$JSONL" ]; then
    ROWS=$(wc -l < "$JSONL")
    echo "  Text JSONL:   ${ROWS} rows"
else
    echo "  Text JSONL:   not found"
fi
REMOTE_EOF

echo ""
echo "── gunicorn process check ───────────────────────────────"
ssh -i "$KEY_PATH" ubuntu@$EC2_IP \
    "pgrep -a gunicorn || echo '  ⚠️  gunicorn process not found'"

echo ""
echo "── Quick web honeypot HTTP check (from EC2 internal) ────"
ssh -i "$KEY_PATH" ubuntu@$EC2_IP \
    "curl -s http://localhost:5001/health || echo '  ⚠️  Web honeypot /health check failed'"

echo ""
echo "========================================================="
echo "✅ Status check complete."
echo ""
echo "EXTERNAL CHECKS (run these from your laptop separately):"
echo "  curl http://${EC2_IP}/health          # port 80 → 5001 redirect"
echo "  curl http://${EC2_IP}:5001/health     # direct"
echo "  nc -w3 ${EC2_IP} 6379               # Redis port (should connect)"
echo "  nc -w3 ${EC2_IP} 2222               # SSH banner"
echo "========================================================="