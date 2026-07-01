"""
check_collection_progress.py — [Partner B]

Runs on your LAPTOP (not EC2), after pulling data down via infra/pull_data.sh.
Tells you whether you have enough real or synthetic data to start Phase 3
model training, without needing to SSH into the instance.

Usage:
    bash infra/pull_data.sh
    python scripts/check_collection_progress.py

    if aws issue then :
    aws configure

    after putting credentials, run:
    aws s3 sync s3://aml-fintech-honeypot-dataset/data/raw/ ./data/raw/ --exclude "*.tmp"
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

TARGET_ROWS = 50000
SYNTHETIC_MIN_ROWS = 10000
PROGRESS_BAR_WIDTH = 20


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _progress_bar(current: int, target: int, width: int = PROGRESS_BAR_WIDTH) -> str:
    """Builds a string like [████████░░] 42,300 / 50,000 (84.6%)."""
    pct = min(current / target, 1.0) if target > 0 else 0.0
    filled = int(round(pct * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current:,} / {target:,} ({pct * 100:.1f}%)"


def _parse_timestamp(value: str) -> Optional[datetime]:
    """Best-effort ISO 8601 timestamp parsing, tolerant of missing tz info."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _eta_hours(current: int, target: int, hours_elapsed: float) -> float:
    """Hours remaining to reach target, given current rate. inf if rate is 0."""
    if current <= 0 or hours_elapsed <= 0:
        return float("inf")
    rate_per_hour = current / hours_elapsed
    remaining = max(target - current, 0)
    if rate_per_hour <= 0:
        return float("inf")
    return remaining / rate_per_hour


# --------------------------------------------------------------------------- #
# 1. Tabular progress
# --------------------------------------------------------------------------- #
def check_tabular_progress(csv_path: str = "data/raw/tabular/honeypot_log.csv") -> dict:
    """
    Reports collection progress for the network honeypot's tabular data.
    Returns a dict; also prints a human-readable summary with a progress bar.
    """
    path = Path(csv_path)
    result = {
        "exists": path.exists(),
        "total_rows": 0,
        "unique_src_ips": 0,
        "label_distribution": {},
        "collection_rate_per_hour": 0.0,
        "eta_hours_to_target": float("inf"),
        "target_rows": TARGET_ROWS,
    }

    if not path.exists():
        print(f"❌ Tabular data file not found: {csv_path}")
        return result

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"⚠️  Tabular data file exists but is empty: {csv_path}")
        return result

    total_rows = len(df)
    result["total_rows"] = total_rows

    if total_rows == 0:
        print(f"⚠️  Tabular data file has a header but no rows yet: {csv_path}")
        return result

    if "src_ip" in df.columns:
        result["unique_src_ips"] = int(df["src_ip"].nunique())

    if "label" in df.columns:
        result["label_distribution"] = df["label"].value_counts().to_dict()

    # Collection rate based on first vs. most recent timestamp in the data
    hours_elapsed = 0.0
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].apply(_parse_timestamp).dropna()
        if len(timestamps) >= 2:
            span = timestamps.max() - timestamps.min()
            hours_elapsed = span.total_seconds() / 3600.0

    rate = total_rows / hours_elapsed if hours_elapsed > 0 else 0.0
    result["collection_rate_per_hour"] = round(rate, 2)
    result["eta_hours_to_target"] = round(
        _eta_hours(total_rows, TARGET_ROWS, hours_elapsed), 2
    )

    print("\n--- Tabular (Network Honeypot) ---")
    print(_progress_bar(total_rows, TARGET_ROWS))
    print(f"Unique source IPs: {result['unique_src_ips']:,}")
    print(f"Label distribution: {result['label_distribution']}")
    print(f"Collection rate: {result['collection_rate_per_hour']:.1f} rows/hr")
    eta = result["eta_hours_to_target"]
    eta_str = f"{eta:.1f} hrs" if eta != float("inf") else "unknown (insufficient data)"
    print(f"ETA to {TARGET_ROWS:,} rows: {eta_str}")

    return result


# --------------------------------------------------------------------------- #
# 2. Text progress
# --------------------------------------------------------------------------- #
def check_text_progress(jsonl_path: str = "data/raw/text/api_logs.jsonl") -> dict:
    """
    Reports collection progress for the web honeypot's text/API payload data.
    Returns a dict; also prints a human-readable summary.
    """
    path = Path(jsonl_path)
    result = {
        "exists": path.exists(),
        "total_rows": 0,
        "label_0_count": 0,
        "label_1_count": 0,
        "top_endpoints": [],
        "top_user_agents": [],
        "eta_hours_to_target": float("inf"),
        "target_rows": TARGET_ROWS,
    }

    if not path.exists():
        print(f"\n❌ Text data file not found: {jsonl_path}")
        return result

    records = []
    malformed_lines = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed_lines += 1

    total_rows = len(records)
    result["total_rows"] = total_rows

    if malformed_lines > 0:
        print(f"⚠️  Skipped {malformed_lines} malformed JSON line(s) in {jsonl_path}")

    if total_rows == 0:
        print(f"\n⚠️  Text data file exists but has no valid rows yet: {jsonl_path}")
        return result

    labels = [r.get("label") for r in records if "label" in r]
    result["label_0_count"] = labels.count(0)
    result["label_1_count"] = labels.count(1)

    endpoints = [r.get("endpoint", "") for r in records if r.get("endpoint")]
    result["top_endpoints"] = Counter(endpoints).most_common(5)

    user_agents = [r.get("user_agent", "") for r in records if r.get("user_agent")]
    result["top_user_agents"] = Counter(user_agents).most_common(5)

    # ETA based on first vs. most recent timestamp
    hours_elapsed = 0.0
    timestamps = [
        ts for ts in (_parse_timestamp(r.get("timestamp", "")) for r in records) if ts
    ]
    if len(timestamps) >= 2:
        span = max(timestamps) - min(timestamps)
        hours_elapsed = span.total_seconds() / 3600.0

    result["eta_hours_to_target"] = round(
        _eta_hours(total_rows, TARGET_ROWS, hours_elapsed), 2
    )

    print("\n--- Text (Web Honeypot) ---")
    print(_progress_bar(total_rows, TARGET_ROWS))
    print(f"Benign (label=0): {result['label_0_count']:,} | Malicious (label=1): {result['label_1_count']:,}")
    print("Top endpoints hit:")
    for endpoint, count in result["top_endpoints"]:
        print(f"  {endpoint}: {count:,}")
    print("Top user agents:")
    for ua, count in result["top_user_agents"]:
        display_ua = ua if len(ua) <= 60 else ua[:57] + "..."
        print(f"  {display_ua}: {count:,}")
    eta = result["eta_hours_to_target"]
    eta_str = f"{eta:.1f} hrs" if eta != float("inf") else "unknown (insufficient data)"
    print(f"ETA to {TARGET_ROWS:,} rows: {eta_str}")

    return result


# --------------------------------------------------------------------------- #
# 3. Synthetic data readiness
# --------------------------------------------------------------------------- #
def check_synthetic_ready(
    synthetic_tabular_path: str = "data/raw/tabular/synthetic_traffic.csv",
    synthetic_text_path: str = "data/raw/text/synthetic_payloads.jsonl",
    min_rows: int = SYNTHETIC_MIN_ROWS,
) -> bool:
    """
    Checks whether synthetic tabular and text datasets exist and meet the
    minimum row threshold needed to start Phase 3 training.
    """
    tabular_path = Path(synthetic_tabular_path)
    text_path = Path(synthetic_text_path)

    tabular_rows = 0
    text_rows = 0

    if tabular_path.exists():
        try:
            tabular_rows = len(pd.read_csv(tabular_path))
        except pd.errors.EmptyDataError:
            tabular_rows = 0

    if text_path.exists():
        with open(text_path, "r") as f:
            text_rows = sum(1 for line in f if line.strip())

    ready = tabular_rows >= min_rows and text_rows >= min_rows

    print("\n--- Synthetic Data Readiness ---")
    print(f"Synthetic tabular rows: {tabular_rows:,} (need ≥ {min_rows:,})")
    print(f"Synthetic text rows: {text_rows:,} (need ≥ {min_rows:,})")

    if ready:
        print("✅ Synthetic data ready for training")
    else:
        print("❌ Run synthetic generators first")

    return ready


# --------------------------------------------------------------------------- #
# 4. Full report
# --------------------------------------------------------------------------- #
def print_full_report():
    """
    Runs all three checks and prints a combined summary with a clear
    recommendation on whether to proceed to Phase 3 now (using synthetic
    data) or wait for real collection to finish.
    """
    print("=" * 60)
    print("AML FinTech Framework — Data Collection Progress Report")
    print("=" * 60)

    tabular_result = check_tabular_progress()
    text_result = check_text_progress()
    synthetic_ready = check_synthetic_ready()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"{'Modality':<12} {'Rows':>10} {'Target':>10} {'Status':<20}"
    )
    print("-" * 60)
    tabular_rows = tabular_result["total_rows"]
    text_rows = text_result["total_rows"]
    print(
        f"{'Tabular':<12} {tabular_rows:>10,} {TARGET_ROWS:>10,} "
        f"{'✅ Complete' if tabular_rows >= TARGET_ROWS else '⏳ Collecting':<20}"
    )
    print(
        f"{'Text':<12} {text_rows:>10,} {TARGET_ROWS:>10,} "
        f"{'✅ Complete' if text_rows >= TARGET_ROWS else '⏳ Collecting':<20}"
    )
    print(
        f"{'Synthetic':<12} {'n/a':>10} {'n/a':>10} "
        f"{'✅ Ready' if synthetic_ready else '❌ Not ready':<20}"
    )

    print()
    both_complete = tabular_rows >= TARGET_ROWS and text_rows >= TARGET_ROWS
    either_incomplete = tabular_rows < TARGET_ROWS or text_rows < TARGET_ROWS

    if both_complete:
        print("🎉 Collection complete! Run post-collection pipeline before Phase 3.")
    elif either_incomplete:
        print("⚠️  Real data still collecting. Use synthetic data for Phase 3 training.")
        if synthetic_ready:
            print("✅  You can start Phase 3 now with synthetic data.")
        else:
            print("❌  Synthetic data isn't ready yet either — run the generators first")
            print("    (scripts/generate_synthetic_tabular.py and generate_synthetic_text.py).")


if __name__ == "__main__":
    print_full_report()