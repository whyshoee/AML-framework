#!/bin/bash
# infra/pull_data.sh  — [Partner B]
#
# Run from your LAPTOP once a day (or whenever you want a fresh snapshot)
# to pull collected honeypot data from S3 down to ./data/raw/.
#
# BUGS FIXED vs original:
#   1. BUCKET_NAME was set to the literal string "REPLACE_WITH_YOUR_BUCKET"
#      (never updated). Now set to the actual bucket name.
#   2. Added --exact-timestamps to preserve original file modification times.
#   3. Added row counts for all new data files (tabular + text).
#   4. Added a one-line per-port breakdown if service_type column exists in CSV.
#
# USAGE:
#   bash infra/pull_data.sh
#
# AFTER PULLING:
#   python scripts/check_collection_progress.py

set -e

BUCKET_NAME="aml-fintech-honeypot-dataset"

echo "========================================================="
echo "AML FinTech — Pull Data from S3"
echo "Bucket: s3://${BUCKET_NAME}"
echo "========================================================="

echo ""
echo "Syncing s3://${BUCKET_NAME}/data/raw/ → ./data/raw/ ..."
aws s3 sync "s3://${BUCKET_NAME}/data/raw/" "./data/raw/" \
    --exclude "*.tmp" \
    --exact-timestamps

echo ""
echo "── Row counts ───────────────────────────────────────────"

TABULAR_CSV="data/raw/tabular/honeypot_log.csv"
TEXT_JSONL="data/raw/text/api_logs.jsonl"

if [ -f "$TABULAR_CSV" ]; then
    # Subtract 1 for the header row
    ROWS=$(( $(wc -l < "$TABULAR_CSV") - 1 ))
    echo "  Tabular (network honeypot):  ${ROWS} rows  ($TABULAR_CSV)"

    # If the new 16-column schema is present, show per-service_type breakdown
    # Uses python3 for the column parse since awk can't reliably handle quoted CSV
    if python3 -c "
    
import csv, sys
from collections import Counter
with open('${TABULAR_CSV}') as f:
    reader = csv.DictReader(f)
    if 'service_type' in reader.fieldnames:
        c = Counter(row['service_type'] for row in reader)
        print('  Per-port breakdown:')
        for stype, n in sorted(c.items(), key=lambda x: -x[1]):
            print(f'    {stype:<12} {n:>8,} rows')
    else:
        print('  (old 15-col schema — service_type not yet present)')
" 2>/dev/null; then
        :
    fi
else
    echo "  Tabular (network honeypot):  no file yet  ($TABULAR_CSV)"
fi

if [ -f "$TEXT_JSONL" ]; then
    ROWS=$(wc -l < "$TEXT_JSONL")
    echo "  Text (web honeypot):         ${ROWS} rows  ($TEXT_JSONL)"
else
    echo "  Text (web honeypot):         no file yet  ($TEXT_JSONL)"
fi

# Also check synthetic data if already generated
SYNTH_CSV="data/raw/tabular/synthetic_traffic.csv"
SYNTH_JSONL="data/raw/text/synthetic_payloads.jsonl"

if [ -f "$SYNTH_CSV" ]; then
    ROWS=$(( $(wc -l < "$SYNTH_CSV") - 1 ))
    echo "  Synthetic tabular:           ${ROWS} rows  ($SYNTH_CSV)"
fi
if [ -f "$SYNTH_JSONL" ]; then
    ROWS=$(wc -l < "$SYNTH_JSONL")
    echo "  Synthetic text:              ${ROWS} rows  ($SYNTH_JSONL)"
fi

echo ""
echo "── Collection progress ──────────────────────────────────"
python3 scripts/check_collection_progress.py

echo ""
echo "========================================================="
echo "✅ Pull complete."
echo "   Next: python scripts/check_collection_progress.py"
echo "========================================================="