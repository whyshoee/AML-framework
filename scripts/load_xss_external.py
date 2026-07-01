"""
[Partner B]
scripts/load_xss_external.py

Downloads a real, independently-collected, peer-reviewed XSS attack dataset
and reshapes it to match the (payload, label) schema your DistilBERT model
expects, so you can sanity-check the model against text that did NOT come
from your own synthetic_payloads.jsonl templates.

WHY THIS MATTERS: your DistilBERT model hit eval_accuracy=1.0 / eval_f1=1.0
on synthetic_payloads.jsonl. A naive keyword-match rule (looking for
"<script", "OR '1'='1", etc.) ALSO gets ~85%+ on that same data with zero
machine learning at all -- meaning the synthetic generator's attack
templates are too mechanically distinct from its benign templates for a
99-100% score to mean much. This script provides an external, real-world
check: payloads nobody designed to be easy or hard for your specific model.

SOURCE & CITATION:
Mereani, F. A. and Howe, J. M. (2018). "Detecting Cross-Site Scripting
Attacks Using Machine Learning." In Advanced Machine Learning Technologies
and Applications, vol. 723 of AISC, pp. 200-210. Springer.
Mereani, F. A. and Howe, J. M. (2018). "Preventing Cross-Site Scripting
Attacks by Combining Classifiers." Proceedings of IJCCI 2018, Vol 1,
pp. 135-143. SciTePress.
Repository: github.com/fmereani/Cross-Site-Scripting-XSS

IMPORTANT SCHEMA CAVEAT (read before drawing conclusions):
This dataset's "payloads" are scraped URLs/query-strings from real
XSS-vulnerable web pages -- not JSON API request bodies like your
honeypot collects. It is a genuine, real-world, human-attacker-authored
text classification task (URL string -> malicious/benign), which makes it
a meaningful "does the model generalize beyond its own synthetic
templates" check -- but it is NOT a perfect structural match to your
web_honeypot.py's JSON payload format. Treat results as informative about
generalization, not as a literal apples-to-apples FinTech API benchmark.

Run:
    python scripts/load_xss_external.py
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("load_xss_external")

SOURCE_URL = (
    "https://raw.githubusercontent.com/fmereani/Cross-Site-Scripting-XSS/"
    "master/XSSDataSets/Payloads.csv"
)

# Saved under a clearly separate path from BOTH your synthetic generator's
# output (data/raw/text/synthetic_payloads.jsonl) and the NSL-KDD tabular
# external set (data/external/tabular/), so nothing collides:
#   data/external/text/xss_mapped.jsonl
OUTPUT_DIR = Path("data/external/text")
OUTPUT_FILENAME = "xss_mapped.jsonl"
RAW_CACHE_FILENAME = "xss_raw_cache.csv"


def download_xss_dataset(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / RAW_CACHE_FILENAME

    if cache_path.exists():
        logger.info("Using cached raw file at %s (delete it to force re-download)", cache_path)
        return cache_path

    logger.info("Downloading XSS dataset from %s", SOURCE_URL)
    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    logger.info("Saved raw file (%d bytes) to %s", len(response.content), cache_path)
    return cache_path


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    """
    The source CSV has occasional non-UTF-8 bytes and a small number of
    malformed rows (verified by testing -- ~99 of 43316 rows fail to
    parse). latin-1 encoding accepts any byte value and on_bad_lines="skip"
    drops the unparseable rows rather than crashing.
    """
    df = pd.read_csv(raw_path, on_bad_lines="skip", engine="python", encoding="latin-1")
    df = df.dropna(subset=["Payloads", "Class"])
    logger.info("Loaded %d valid rows after cleaning", len(df))
    return df


def map_to_payload_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map the source's (Payloads, Class) columns onto your (payload, label) schema."""
    out = pd.DataFrame()
    out["payload"] = df["Payloads"].astype(str)
    # Source uses string labels "Malicious" / "Benign" -- map to your binary
    # int schema (0=benign, 1=attack), same convention as synthetic_payloads.jsonl.
    out["label"] = (df["Class"].str.strip().str.lower() == "malicious").astype(int)
    out["endpoint"] = "external/unknown"  # source has no endpoint field; flagged as such
    out["method"] = "GET"  # these are scraped GET-request URLs, not POST JSON bodies
    out["attack_type"] = out["label"].map({0: "benign", 1: "xss_external"})
    return out


def validate_mapped_dataset(df: pd.DataFrame) -> None:
    if df["payload"].isnull().any() or (df["payload"].str.len() == 0).any():
        raise ValueError("Mapped dataset contains null or empty payloads.")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Label column is not binary.")
    logger.info("Validation passed: no null/empty payloads, binary label confirmed.")


def main():
    parser = argparse.ArgumentParser(
        description="Download a real-world XSS dataset and reshape it for DistilBERT cross-testing."
    )
    parser.add_argument("--cache-dir", default="data/external/text/_raw_cache")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Optional cap for a quick test run (e.g. --max-rows 5000).",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = download_xss_dataset(cache_dir)
    df_raw = load_and_clean(raw_path)

    if args.max_rows:
        df_raw = df_raw.sample(n=min(args.max_rows, len(df_raw)), random_state=42)
        logger.info("Sub-sampled to %d rows via --max-rows", len(df_raw))

    df_mapped = map_to_payload_schema(df_raw)
    validate_mapped_dataset(df_mapped)

    output_path = output_dir / OUTPUT_FILENAME
    with open(output_path, "w") as f:
        for _, row in df_mapped.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    logger.info("Saved mapped dataset (%d rows) to %s", len(df_mapped), output_path)

    stats = {
        "source": "XSS Dataset (Mereani & Howe, 2018)",
        "source_url": SOURCE_URL,
        "citation": [
            "Mereani, F. A. and Howe, J. M. (2018). Detecting Cross-Site Scripting "
            "Attacks Using Machine Learning. Advanced Machine Learning Technologies "
            "and Applications, AISC vol. 723, pp. 200-210. Springer.",
            "Mereani, F. A. and Howe, J. M. (2018). Preventing Cross-Site Scripting "
            "Attacks by Combining Classifiers. Proceedings of IJCCI 2018, Vol 1, "
            "pp. 135-143. SciTePress.",
        ],
        "n_rows": int(len(df_mapped)),
        "label_distribution": df_mapped["label"].value_counts().to_dict(),
        "caveat": (
            "Payloads are scraped URLs/query-strings from real XSS-vulnerable pages, "
            "not JSON API request bodies. Useful for testing generalization beyond "
            "synthetic_payloads.jsonl's templates, NOT a literal structural match to "
            "web_honeypot.py's payload format."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    stats_path = output_dir / "xss_mapped_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved dataset card to %s", stats_path)

    print(f"\n✅ External text validation dataset ready: {len(df_mapped)} rows")
    print(f"   Saved to: {output_path}")
    print(f"   (Does NOT overwrite data/raw/text/synthetic_payloads.jsonl)")
    print(f"\nLabel distribution: {df_mapped['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
