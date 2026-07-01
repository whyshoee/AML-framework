"""
[Partner B]
train_distilbert.py — Fine-tunes DistilBERT for binary classification on
JSON API payload text (benign vs. attack) for the AML FinTech Framework.

Pipeline:
    1. Load synthetic_payloads.jsonl (payload, label per line).
    2. Tokenize with distilbert-base-uncased, max_length=128.
    3. 70/15/15 stratified split.
    4. Fine-tune via HuggingFace Trainer, 5 epochs, best checkpoint by F1.
    5. Evaluate on the held-out test set, save metrics.
    6. Save best model + tokenizer to models/distilbert/best/.
    7. Expose classify_payload() for downstream use (Phase 6 API / attacks).

Run:
    python scripts/train_distilbert.py
    python scripts/train_distilbert.py --epochs 3 --data-path data/processed/text/cleaned_final.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_distilbert")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


# ---------------------------------------------------------------------------
# Device handling — CUDA > MPS > CPU, matching the rest of the framework's
# convention (mirrors train_resnet.py's auto-select logic).
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_payloads(jsonl_path: str) -> pd.DataFrame:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run scripts/generate_synthetic_text.py first, "
            "or pass --data-path pointing at the merged/cleaned dataset."
        )

    records = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON on line %d", line_no)
                continue
            payload = obj.get("payload")
            label = obj.get("label")
            if payload is None or label is None:
                continue
            # The "payload" field may already be a JSON string (most common,
            # since the generator dumps a dict via json.dumps) or, less
            # commonly, an actual nested dict if the JSONL was hand-built.
            # Normalize both cases to a single string for tokenization.
            if isinstance(payload, (dict, list)):
                payload_str = json.dumps(payload)
            else:
                payload_str = str(payload)
            records.append({"payload": payload_str, "label": int(label)})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid records loaded from {path}.")
    logger.info("Loaded %d payload records from %s", len(df), path)
    logger.info("Label distribution: %s", df["label"].value_counts().to_dict())
    return df


def split_data(df: pd.DataFrame, seed: int = 42):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df)
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class APIPayloadDataset(Dataset):
    """Tokenizes payload strings on the fly and returns tensors HF Trainer expects."""

    def __init__(self, payloads, labels, tokenizer, max_length: int = MAX_LENGTH):
        self.payloads = list(payloads)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.payloads)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.payloads[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Inference helper — importable by Phase 4 attacks / Phase 6 FastAPI backend
# ---------------------------------------------------------------------------
def classify_payload(
    payload_str: str,
    model_dir: str = "models/distilbert/best",
    max_length: int = MAX_LENGTH,
) -> dict:
    """
    Run inference on a single raw payload string and return
    {"label": 0 or 1, "prob": probability of the predicted class}.

    Loads the model fresh each call -- fine for the API/attack use cases
    this is designed for (low QPS, correctness over latency). If this needs
    to be called in a tight loop, load the model/tokenizer once outside and
    reuse rather than calling this function repeatedly.
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    encoding = tokenizer(
        payload_str,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_label = int(torch.argmax(probs).item())
        pred_prob = float(probs[pred_label].item())

    return {"label": pred_label, "prob": pred_prob}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on API payload data.")
    parser.add_argument(
        "--data-path", type=str, default="data/raw/text/synthetic_payloads.jsonl",
        help="Path to the input JSONL (synthetic or merged/cleaned real+synthetic).",
    )
    parser.add_argument("--output-dir", type=str, default="models/distilbert")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best"

    device = get_device()

    df = load_payloads(args.data_path)
    train_df, val_df, test_df = split_data(df, seed=args.seed)

    logger.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_dataset = APIPayloadDataset(train_df["payload"], train_df["label"], tokenizer)
    val_dataset = APIPayloadDataset(val_df["payload"], val_df["label"], tokenizer)
    test_dataset = APIPayloadDataset(test_df["payload"], test_df["label"], tokenizer)

    # NOTE on API drift: the spec's TrainingArguments used
    # evaluation_strategy="epoch", which was renamed to eval_strategy in
    # transformers' 4.x->5.x line (confirmed against the installed version
    # before writing this -- evaluation_strategy raises a TypeError now).
    # Corrected below so this script actually runs rather than crashing on
    # the first line of training setup.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],  # disable wandb/tensorboard auto-detection in this environment
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training: %d epochs, %d train examples", args.epochs, len(train_dataset))
    trainer.train()

    logger.info("Evaluating on held-out test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    logger.info("Test metrics: %s", test_results)

    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    logger.info("Saved metrics to %s", metrics_path)

    best_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(best_dir))
    trainer.model.save_pretrained(str(best_dir))
    logger.info("Saved best tokenizer + model to %s", best_dir)

    logger.info("✅ DistilBERT training complete. Test F1: %.4f", test_results.get("eval_f1", -1))


if __name__ == "__main__":
    main()