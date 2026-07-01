"""
migrate_existing_data.py — [Partner B]

Run this ONCE on your LAPTOP before deploying the upgraded network_honeypot.py.

WHY THIS EXISTS:
  The upgraded network_honeypot.py adds a 'service_type' column to the CSV
  (e.g. "ssh_alt", "telnet", "redis"). Your existing honeypot_log.csv has
  15 columns without it. This script patches the old data in-place so that:
    - All old rows get service_type = "custom"  (they all came from port 9999)
    - The file becomes 16-column schema compatible with the new honeypot
    - No rows are lost — existing collected data is fully preserved

WHAT IT DOES:
  1. Creates a timestamped backup of your original CSV first (non-destructive).
  2. Reads the original CSV into pandas.
  3. Inserts 'service_type' = "custom" in the correct column position.
  4. Writes the migrated CSV back to the same path.
  5. Validates the result and prints a row-count check.

ALSO HANDLES:
  - api_logs.jsonl: no schema change needed (web honeypot JSONL is unchanged).
  - Empty or missing files: handled gracefully with a warning.

RUN:
  python scripts/migrate_existing_data.py
  # or with explicit paths:
  python scripts/migrate_existing_data.py \
      --csv  data/raw/tabular/honeypot_log.csv \
      --jsonl data/raw/text/api_logs.jsonl
"""

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# The new 16-column schema — service_type is at position 3 (after dst_port)
NEW_CSV_FIELDNAMES = [
    "src_ip", "src_port", "dst_port", "service_type",
    "timestamp", "unix_timestamp",
    "bytes_received", "connection_duration_ms",
    "payload_entropy", "packet_size",
    "is_repeated_src", "src_ip_frequency",
    "payload_printable_ratio", "payload_hex_preview",
    "tcp_flags_estimated", "label",
]

# All existing rows came from port 9999 (the original single-port honeypot)
DEFAULT_SERVICE_TYPE_FOR_OLD_DATA = "custom"


def backup_file(path: Path) -> Path:
    """Copies path → path.bak.YYYYMMDD_HHMMSS. Returns the backup path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(f".bak.{ts}")
    shutil.copy2(path, backup)
    print(f"  ✅ Backup created: {backup}")
    return backup


def migrate_tabular_csv(csv_path: Path) -> None:
    """
    Adds 'service_type' column to the existing honeypot_log.csv.
    All old rows are stamped with service_type = "custom" (port 9999 traffic).
    """
    print(f"\n── Tabular CSV migration: {csv_path} ──")

    if not csv_path.exists():
        print("  ⚠️  File not found — nothing to migrate.")
        return

    if csv_path.stat().st_size == 0:
        print("  ⚠️  File is empty — writing fresh 16-column header only.")
        csv_path.write_text(",".join(NEW_CSV_FIELDNAMES) + "\n")
        return

    # Load existing data
    df = pd.read_csv(csv_path)
    original_rows = len(df)
    print(f"  Loaded {original_rows:,} rows, {len(df.columns)} columns.")

    # Check if already migrated
    if "service_type" in df.columns:
        print("  ✅ 'service_type' column already present — no migration needed.")
        # Verify all fieldnames match
        missing = [f for f in NEW_CSV_FIELDNAMES if f not in df.columns]
        if missing:
            print(f"  ⚠️  Unexpected schema — missing columns: {missing}")
        return

    # Create backup before modifying
    backup_file(csv_path)

    # Insert 'service_type' after 'dst_port' (position 3 in the new schema)
    # pandas insert(loc, column, value) inserts before the column at `loc`
    insert_position = df.columns.tolist().index("dst_port") + 1
    df.insert(
        loc=insert_position,
        column="service_type",
        value=DEFAULT_SERVICE_TYPE_FOR_OLD_DATA,
    )
    print(
        f"  Inserted 'service_type' = \"{DEFAULT_SERVICE_TYPE_FOR_OLD_DATA}\" "
        f"at column position {insert_position}."
    )

    # Reorder to exactly match the new schema (in case old file had different column order)
    # Only include columns that exist in both old data and new schema
    available = [c for c in NEW_CSV_FIELDNAMES if c in df.columns]
    extra = [c for c in df.columns if c not in NEW_CSV_FIELDNAMES]
    if extra:
        print(f"  ⚠️  Old CSV had extra columns not in new schema: {extra} — dropping them.")
    df = df[available]

    # Validate no rows were lost
    assert len(df) == original_rows, (
        f"Row count changed during migration! Before: {original_rows}, After: {len(df)}"
    )

    # Write migrated CSV
    df.to_csv(csv_path, index=False)
    print(
        f"  ✅ Migration complete: {len(df):,} rows × {len(df.columns)} columns "
        f"written to {csv_path}"
    )

    # Sanity check: reload and verify
    check = pd.read_csv(csv_path)
    assert len(check) == original_rows, "Row count mismatch after write!"
    assert "service_type" in check.columns, "'service_type' column missing after write!"
    assert list(check.columns) == available, "Column order mismatch after write!"
    print(f"  ✅ Verification passed: {len(check):,} rows, columns match new schema.")


def check_jsonl(jsonl_path: Path) -> None:
    """
    Checks the api_logs.jsonl for any issues.
    No schema migration needed — web honeypot JSONL format is unchanged.
    """
    print(f"\n── Text JSONL check: {jsonl_path} ──")

    if not jsonl_path.exists():
        print("  ⚠️  File not found — nothing to check.")
        return

    valid = 0
    bad = 0
    with open(jsonl_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                import json
                json.loads(line)
                valid += 1
            except Exception:
                bad += 1
                if bad <= 3:
                    print(f"  ⚠️  Line {i} is not valid JSON: {line[:80]}...")

    print(f"  Valid JSON lines: {valid:,}")
    if bad > 0:
        print(f"  ⚠️  Malformed lines: {bad:,} (these will be skipped by pandas / training)")
    else:
        print(f"  ✅ All lines are valid JSON — no migration needed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate existing honeypot data to the new 16-column schema."
    )
    parser.add_argument(
        "--csv",
        default="data/raw/tabular/honeypot_log.csv",
        help="Path to honeypot_log.csv (default: data/raw/tabular/honeypot_log.csv)",
    )
    parser.add_argument(
        "--jsonl",
        default="data/raw/text/api_logs.jsonl",
        help="Path to api_logs.jsonl (default: data/raw/text/api_logs.jsonl)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AML FinTech — Existing Data Migration")
    print("=" * 60)
    print("This script is safe to re-run. It will skip files that are")
    print("already in the new 16-column schema.\n")

    migrate_tabular_csv(Path(args.csv))
    check_jsonl(Path(args.jsonl))

    print("\n" + "=" * 60)
    print("Migration complete.")
    print(
        "Next steps:\n"
        "  1. Run infra/deploy_honeypots.sh to push the upgraded honeypot scripts\n"
        "  2. Also push this migrated CSV to EC2 if the instance still has old data:\n"
        "       scp -i ~/.ssh/aml-fintech-key.pem \\\n"
        "           data/raw/tabular/honeypot_log.csv \\\n"
        "           ubuntu@16.171.191.100:~/aml_fintech/data/raw/tabular/\n"
        "  3. The new honeypot will append new 16-column rows to the same file."
    )


if __name__ == "__main__":
    main()