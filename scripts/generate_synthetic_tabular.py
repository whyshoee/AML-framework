"""
generate_synthetic_tabular.py — [Partner B]

Generates a high-quality synthetic network traffic dataset that mimics the
real network honeypot's collected schema (see honeypots/network_honeypot.py
CSV_FIELDNAMES). Used to train XGBoost (Phase 3) while real honeypot data
collects on AWS in the background, and later merged with real data
(Phase 2 Task 7) for fine-tuning.

Schema matches network_honeypot.py's 15 features exactly, so a model
trained on this synthetic data can be evaluated/fine-tuned on real data
without any feature-engineering mismatch.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "src_ip", "src_port", "dst_port", "bytes_received",
    "connection_duration_ms", "payload_entropy", "packet_size",
    "is_repeated_src", "src_ip_frequency", "payload_printable_ratio",
    "tcp_flags_estimated", "label",
]

FLOAT_COLUMNS = [
    "bytes_received", "connection_duration_ms", "payload_entropy",
    "packet_size", "src_ip_frequency", "payload_printable_ratio",
]


class SyntheticNetworkTrafficGenerator:
    def __init__(self, seed: int = 42, n_rows: int = 50000, fraud_rate: float = 0.45):
        self.seed = seed
        self.n_rows = n_rows
        self.fraud_rate = fraud_rate
        self.rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------- #
    # IP generation helpers
    # ----------------------------------------------------------------- #
    def _random_rfc1918_ip(self, n: int) -> np.ndarray:
        """Mix of 10.x.x.x and 192.168.x.x private-range IPs."""
        choice = self.rng.integers(0, 2, size=n)
        ips = np.empty(n, dtype=object)
        for i in range(n):
            if choice[i] == 0:
                ips[i] = f"10.{self.rng.integers(0, 256)}.{self.rng.integers(0, 256)}.{self.rng.integers(1, 255)}"
            else:
                ips[i] = f"192.168.{self.rng.integers(0, 256)}.{self.rng.integers(1, 255)}"
        return ips

    def _random_public_ip(self, n: int) -> np.ndarray:
        """Random public-range IPs (avoiding reserved/private blocks)."""
        ips = np.empty(n, dtype=object)
        for i in range(n):
            first = self.rng.integers(1, 224)
            while first in (10, 127, 169, 172, 192):  # skip private/reserved-ish ranges
                first = self.rng.integers(1, 224)
            ips[i] = f"{first}.{self.rng.integers(0, 256)}.{self.rng.integers(0, 256)}.{self.rng.integers(1, 255)}"
        return ips

    def _benign_src_ips(self, n: int) -> np.ndarray:
        """80% RFC1918 (internal scanners), 20% known-research-style public IPs."""
        n_private = int(round(n * 0.8))
        n_public = n - n_private
        private_ips = self._random_rfc1918_ip(n_private)
        public_ips = self._random_public_ip(n_public)
        combined = np.concatenate([private_ips, public_ips])
        self.rng.shuffle(combined)
        return combined

    # ----------------------------------------------------------------- #
    # Benign flow generation
    # ----------------------------------------------------------------- #
    def generate_benign_flows(self, n: int) -> pd.DataFrame:
        src_ip = self._benign_src_ips(n)
        src_port = self.rng.integers(32768, 61000, size=n)
        dst_port = np.full(n, 9999)

        bytes_received = np.clip(self.rng.normal(256, 64, size=n), 0, 1024)
        connection_duration_ms = np.clip(self.rng.exponential(200, size=n), 1, 5000)
        payload_entropy = np.clip(self.rng.normal(3.5, 0.8, size=n), 0, 8)
        packet_size = np.clip(self.rng.normal(180, 60, size=n), 0, 1024)
        is_repeated_src = self.rng.random(n) < 0.1
        src_ip_frequency = self.rng.poisson(2, size=n)
        payload_printable_ratio = np.clip(self.rng.normal(0.85, 0.1, size=n), 0, 1)

        tcp_flags = np.where(self.rng.random(n) < 0.9, "DATA", "SYN_ONLY")

        df = pd.DataFrame({
            "src_ip": src_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "bytes_received": bytes_received,
            "connection_duration_ms": connection_duration_ms,
            "payload_entropy": payload_entropy,
            "packet_size": packet_size,
            "is_repeated_src": is_repeated_src,
            "src_ip_frequency": src_ip_frequency,
            "payload_printable_ratio": payload_printable_ratio,
            "tcp_flags_estimated": tcp_flags,
            "label": 0,
        })
        return df

    # ----------------------------------------------------------------- #
    # Attack flow generation — five archetypes
    # ----------------------------------------------------------------- #
    def _archetype_a_port_scanner(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "bytes_received": np.zeros(n),
            "connection_duration_ms": self.rng.uniform(1, 50, size=n),
            "payload_entropy": np.zeros(n),
            "packet_size": np.zeros(n),
            "is_repeated_src": np.full(n, True),
            "src_ip_frequency": self.rng.uniform(50, 500, size=n),
            "payload_printable_ratio": np.zeros(n),
            "tcp_flags_estimated": np.full(n, "SYN_ONLY"),
        })

    def _archetype_b_mirai_botnet(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "bytes_received": self.rng.uniform(4, 32, size=n),
            "connection_duration_ms": self.rng.uniform(100, 800, size=n),
            "payload_entropy": self.rng.uniform(0, 1.5, size=n),
            "packet_size": self.rng.uniform(4, 32, size=n),
            "is_repeated_src": self.rng.random(n) < 0.7,
            "src_ip_frequency": self.rng.uniform(10, 200, size=n),
            "payload_printable_ratio": self.rng.uniform(0.0, 0.3, size=n),
            "tcp_flags_estimated": np.full(n, "DATA"),
        })

    def _archetype_c_exploit_kit(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "bytes_received": self.rng.uniform(500, 1024, size=n),
            "connection_duration_ms": self.rng.uniform(200, 2000, size=n),
            "payload_entropy": self.rng.uniform(6.5, 8.0, size=n),
            "packet_size": self.rng.uniform(500, 1024, size=n),
            "is_repeated_src": self.rng.random(n) < 0.3,
            "src_ip_frequency": self.rng.uniform(1, 50, size=n),
            "payload_printable_ratio": self.rng.uniform(0.1, 0.4, size=n),
            "tcp_flags_estimated": np.full(n, "DATA"),
        })

    def _archetype_d_ssh_bruteforce(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "bytes_received": self.rng.uniform(32, 64, size=n),
            "connection_duration_ms": self.rng.uniform(500, 3000, size=n),
            "payload_entropy": self.rng.uniform(3.0, 4.5, size=n),
            "packet_size": self.rng.uniform(32, 64, size=n),
            "is_repeated_src": self.rng.random(n) < 0.9,
            "src_ip_frequency": self.rng.uniform(100, 1000, size=n),
            "payload_printable_ratio": self.rng.uniform(0.6, 0.95, size=n),
            "tcp_flags_estimated": np.full(n, "DATA"),
        })

    def _archetype_e_credential_stuffing(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "bytes_received": self.rng.uniform(128, 512, size=n),
            "connection_duration_ms": self.rng.uniform(1000, 5000, size=n),
            "payload_entropy": self.rng.uniform(4.0, 6.0, size=n),
            "packet_size": self.rng.uniform(128, 512, size=n),
            "is_repeated_src": self.rng.random(n) < 0.5,
            "src_ip_frequency": self.rng.uniform(20, 300, size=n),
            "payload_printable_ratio": self.rng.uniform(0.5, 0.9, size=n),
            "tcp_flags_estimated": np.full(n, "DATA"),
        })

    def generate_attack_flows(self, n: int) -> pd.DataFrame:
        archetype_weights = {"A": 0.20, "B": 0.25, "C": 0.20, "D": 0.25, "E": 0.10}
        archetype_counts = {}
        remaining = n
        names = list(archetype_weights.keys())
        for name in names[:-1]:
            count = int(round(n * archetype_weights[name]))
            archetype_counts[name] = count
            remaining -= count
        archetype_counts[names[-1]] = remaining  # last archetype absorbs rounding remainder

        builders = {
            "A": self._archetype_a_port_scanner,
            "B": self._archetype_b_mirai_botnet,
            "C": self._archetype_c_exploit_kit,
            "D": self._archetype_d_ssh_bruteforce,
            "E": self._archetype_e_credential_stuffing,
        }

        frames = []
        for name, count in archetype_counts.items():
            if count <= 0:
                continue
            frame = builders[name](count)
            frame["archetype"] = name
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True)

        # Attacker source IPs: mostly public (real internet scanners), with
        # some repeated IPs to reflect botnets reusing the same source.
        combined["src_ip"] = self._random_public_ip(len(combined))
        combined["src_port"] = self.rng.integers(1024, 65536, size=len(combined))
        combined["dst_port"] = 9999
        combined["label"] = 1

        # Clip to valid ranges consistent with the real honeypot's feature space.
        combined["bytes_received"] = np.clip(combined["bytes_received"], 0, 1024)
        combined["packet_size"] = np.clip(combined["packet_size"], 0, 1024)
        combined["payload_entropy"] = np.clip(combined["payload_entropy"], 0, 8)
        combined["payload_printable_ratio"] = np.clip(combined["payload_printable_ratio"], 0, 1)

        return combined.drop(columns=["archetype"])

    # ----------------------------------------------------------------- #
    # Combined generation
    # ----------------------------------------------------------------- #
    def generate(self, n_total: int = 50000) -> pd.DataFrame:
        n_attack = int(round(n_total * self.fraud_rate))
        n_benign = n_total - n_attack

        benign_df = self.generate_benign_flows(n_benign)
        attack_df = self.generate_attack_flows(n_attack)

        combined = pd.concat([benign_df, attack_df], ignore_index=True)
        combined = combined[FEATURE_COLUMNS]

        # Shuffle deterministically.
        combined = combined.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        # Minor Gaussian noise on float columns for realism — avoids the
        # dataset looking artificially "clean" with suspiciously exact
        # archetype boundaries, which would make the model overfit to
        # synthetic-specific artifacts rather than generalizable patterns.
        for col in FLOAT_COLUMNS:
            noise = self.rng.normal(0, 0.01, size=len(combined))
            combined[col] = combined[col] + noise

        # Re-clip after noise to keep values in valid ranges.
        combined["bytes_received"] = np.clip(combined["bytes_received"], 0, 1024)
        combined["connection_duration_ms"] = np.clip(combined["connection_duration_ms"], 0, 5000)
        combined["payload_entropy"] = np.clip(combined["payload_entropy"], 0, 8)
        combined["packet_size"] = np.clip(combined["packet_size"], 0, 1024)
        combined["src_ip_frequency"] = np.clip(combined["src_ip_frequency"], 0, None).round().astype(int)
        combined["payload_printable_ratio"] = np.clip(combined["payload_printable_ratio"], 0, 1)
        combined["is_repeated_src"] = combined["is_repeated_src"].astype(bool)

        return combined

    # ----------------------------------------------------------------- #
    # Save
    # ----------------------------------------------------------------- #
    def save(self, df: pd.DataFrame, output_path: str = "data/raw/tabular/synthetic_traffic.csv"):
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        label_dist = df["label"].value_counts().to_dict()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_means_per_class = (
            df.groupby("label")[numeric_cols].mean().to_dict()
        )

        print(f"\nShape: {df.shape}")
        print(f"Label distribution: {label_dist}")
        print("Feature means per class:")
        for col, means in feature_means_per_class.items():
            print(f"  {col}: {means}")

        stats = {
            "shape": list(df.shape),
            "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
            "feature_means_per_class": {
                str(col): {str(k): float(v) for k, v in means.items()}
                for col, means in feature_means_per_class.items()
            },
        }
        stats_path = out_path.parent / "synthetic_traffic_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to: {stats_path}")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_dataset(df: pd.DataFrame) -> dict:
    results = {}

    nan_count = df.isna().sum().sum()
    results["no_nan_values"] = nan_count == 0

    unique_labels = set(df["label"].unique())
    results["label_is_binary"] = unique_labels.issubset({0, 1})

    range_checks = {
        "src_port": (1024, 65535),
        "dst_port": (9999, 9999),
        "bytes_received": (0, 1024),
        "connection_duration_ms": (0, 5000),
        "payload_entropy": (0, 8),
        "packet_size": (0, 1024),
        "payload_printable_ratio": (0, 1),
    }
    ranges_ok = True
    for col, (lo, hi) in range_checks.items():
        if col not in df.columns:
            continue
        col_min, col_max = df[col].min(), df[col].max()
        # Small tolerance for floating point noise added during generation.
        if col_min < lo - 1 or col_max > hi + 1:
            ranges_ok = False
    results["numeric_features_in_range"] = ranges_ok

    label_props = df["label"].value_counts(normalize=True)
    class_balance_ok = all(0.35 <= p <= 0.65 for p in label_props)
    results["class_balance_within_35_65"] = class_balance_ok

    print("\n--- Validation Report ---")
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    return results


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic network traffic dataset")
    parser.add_argument("--n-rows", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="data/raw/tabular")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraud-rate", type=float, default=0.45)
    args = parser.parse_args()

    generator = SyntheticNetworkTrafficGenerator(
        seed=args.seed, n_rows=args.n_rows, fraud_rate=args.fraud_rate
    )
    df = generator.generate(n_total=args.n_rows)

    validation_results = validate_dataset(df)
    if not all(validation_results.values()):
        print("\n⚠️  One or more validation checks failed — review before training.")

    output_path = str(Path(args.output_dir) / "synthetic_traffic.csv")
    generator.save(df, output_path=output_path)

    print(f"\n✅ Synthetic tabular dataset ready: {len(df)} rows saved to {output_path}")


if __name__ == "__main__":
    main()