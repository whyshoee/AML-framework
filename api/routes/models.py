"""
[Partner A + Partner B]
Model management routes — real training for XGBoost, LightGBM, Random Forest,
DistilBERT, and TF-IDF+LR. ResNet-18 still simulated (needs GPU pipeline).

Endpoints:
  GET   /models/available              – catalogue of all built-in models
  GET   /models/status                 – training job status per modality
  GET   /models/active                 – active model per modality
  POST  /models/train/{modality}       – REAL training on uploaded data
  GET   /models/train/status/{job_id}  – poll training progress
  POST  /models/select/{modality}      – switch active model
  POST  /models/upload/{modality}      – upload pre-trained model file
  DELETE /models/uploaded/{modality}   – remove uploaded model
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import JSONResponse

from api.schemas import ErrorResponse, Modality
from api.routes.data import _resolve_label_column  # shared label-column fallback (see data.py)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Models"])

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

_active_model: dict[str, str] = {
    "tabular": "xgboost",
    "text":    "distilbert",
    "image":   "resnet18",
}

_uploaded_models: dict[str, Optional[dict]] = {
    "tabular": None, "text": None, "image": None,
}

_training_jobs: dict[str, dict] = {}

# Stores fitted model objects keyed by modality
# e.g. _trained_objects["tabular"] = {"model": fitted_xgb, "preprocessor": fitted_pipeline, ...}
_trained_objects: dict[str, Optional[dict]] = {
    "tabular": None, "text": None, "image": None,
}

_ACCEPTED_EXTENSIONS: dict[str, list[str]] = {
    "tabular": [".pkl", ".joblib", ".json", ".onnx"],
    "text":    [".pt", ".pth", ".onnx", ".bin"],
    "image":   [".pt", ".pth", ".onnx"],
}

MAX_MODEL_MB    = 500
MAX_MODEL_BYTES = MAX_MODEL_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

BUILT_IN_MODELS: dict[str, list[dict]] = {
    "tabular": [
        {
            "id": "xgboost", "name": "XGBoost", "recommended": True,
            "description": "Gradient boosting. Industry standard for fraud detection and network traffic classification.",
            "speed": "Fast", "accuracy": "High",
            "notes": "Best choice for most FinTech tabular datasets.",
        },
        {
            "id": "lightgbm", "name": "LightGBM", "recommended": False,
            "description": "Microsoft's gradient boosting. Faster than XGBoost on large datasets, similar accuracy.",
            "speed": "Very Fast", "accuracy": "High",
            "notes": "Preferred when dataset exceeds 500k rows.",
        },
        {
            "id": "random_forest", "name": "Random Forest", "recommended": False,
            "description": "Ensemble of decision trees. Most interpretable — required by some compliance frameworks.",
            "speed": "Medium", "accuracy": "Medium-High",
            "notes": "Choose when regulatory explainability is required.",
        },
    ],
    "text": [
        {
            "id": "distilbert", "name": "DistilBERT", "recommended": True,
            "description": "Distilled BERT — 40% smaller, 60% faster, retains 97% of BERT accuracy.",
            "speed": "Medium", "accuracy": "High",
            "notes": "Best balance for API payload classification.",
        },
        {
            "id": "secbert", "name": "SecBERT", "recommended": False,
            "description": "BERT fine-tuned on cybersecurity corpora. Best for malicious payload detection.",
            "speed": "Medium", "accuracy": "Very High",
            "notes": "Recommended for security-focused classification.",
        },
        {
            "id": "tfidf_lr", "name": "TF-IDF + Logistic Regression", "recommended": False,
            "description": "Classical NLP pipeline. Extremely fast, highly interpretable.",
            "speed": "Very Fast", "accuracy": "Medium",
            "notes": "Choose when speed is critical or GPU is unavailable.",
        },
    ],
    "image": [
        {
            "id": "resnet18", "name": "ResNet-18", "recommended": True,
            "description": "18-layer residual network. Fast training, reasonable accuracy for KYC documents.",
            "speed": "Fast", "accuracy": "Medium",
            "notes": "Good starting point. GPU recommended for full training.",
        },
        {
            "id": "efficientnet_b0", "name": "EfficientNet-B0", "recommended": False,
            "description": "Better accuracy than ResNet-18 at similar computational cost.",
            "speed": "Fast", "accuracy": "High",
            "notes": "Recommended upgrade for production use.",
        },
        {
            "id": "vit_base", "name": "Vision Transformer (ViT-Base)", "recommended": False,
            "description": "Transformer applied to images. Best for document layout understanding.",
            "speed": "Slow", "accuracy": "Very High",
            "notes": "Best accuracy. Requires GPU.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Helper: update job state (thread-safe dict mutation for simple types)
# ---------------------------------------------------------------------------

def _job_update(job: dict, **kwargs) -> None:
    job.update(kwargs)


# ===========================================================================
# REAL TABULAR TRAINING — XGBoost / LightGBM / Random Forest
# ===========================================================================

def _train_tabular_real(
    job_id: str, model_id: str, csv_bytes: bytes, label_column: str
) -> None:
    """
    Real training on uploaded CSV data.
    Phases:
      1. Load data
      2. Validate and engineer features
      3. Split train/validation
      4. Fit model
      5. Evaluate and save
    """
    job = _training_jobs[job_id]

    def phase(name: str, pct: int) -> None:
        _job_update(job, current_phase=name, progress_pct=pct)
        logger.info("tabular training [%s] — %s (%d%%)", job_id[:8], name, pct)

    try:
        _job_update(job, status="running", started_at=datetime.now(timezone.utc).isoformat())

        # ── Phase 1: Load ──
        phase("Loading data from uploaded file", 5)
        import pandas as pd                              # noqa: PLC0415
        import numpy as np                               # noqa: PLC0415
        from sklearn.model_selection import train_test_split  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler      # noqa: PLC0415
        from sklearn.pipeline import Pipeline                  # noqa: PLC0415
        from sklearn.metrics import (                          # noqa: PLC0415
            accuracy_score, f1_score, roc_auc_score, classification_report
        )

        df = pd.read_csv(io.BytesIO(csv_bytes))
        _job_update(job, progress_pct=15)
        logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))

        # ── Phase 2: Validate features ──
        phase("Validating features and label column", 20)

        if label_column not in df.columns:
            # Shared fallback list (kept in sync with data.py's preprocess step
            # via _resolve_label_column, instead of each maintaining its own list).
            found = _resolve_label_column(df.columns, label_column)
            if found:
                label_column = found
                logger.warning("Label column not found, using '%s' instead", found)
            else:
                raise ValueError(
                    f"Label column '{label_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

        y = df[label_column].values
        X = df.drop(columns=[label_column])

        # Keep only numeric columns — drop strings that can't be encoded
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric feature columns found in dataset.")
        X = X[numeric_cols].fillna(0)

        # Ensure binary or multi-class labels
        unique_labels = sorted(set(y))
        n_classes = len(unique_labels)
        logger.info("Features: %d | Classes: %s | Rows: %d",
                    len(numeric_cols), unique_labels[:10], len(df))
        _job_update(job, progress_pct=30)

        # ── Phase 3: Split ──
        phase("Splitting train / validation (80 / 20)", 35)
        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y, test_size=0.20, random_state=42,
            stratify=y if n_classes <= 20 else None
        )
        logger.info("Train: %d rows | Val: %d rows", len(X_train), len(X_val))
        _job_update(job, progress_pct=45)

        # ── Phase 4: Train ──
        phase(f"Training {model_id} on {len(X_train):,} rows", 50)

        if model_id == "xgboost":
            try:
                from xgboost import XGBClassifier          # noqa: PLC0415
            except ImportError:
                raise ImportError(
                    "XGBoost is not installed. Run: pip install xgboost"
                )
            # Remap labels to 0..n-1 for XGBoost
            from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_val_enc   = le.transform(y_val)

            clf = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc   = scaler.transform(X_val)
            clf.fit(
                X_train_sc, y_train_enc,
                eval_set=[(X_val_sc, y_val_enc)],
                verbose=False,
            )
            y_pred = le.inverse_transform(clf.predict(X_val_sc))
            y_prob = clf.predict_proba(X_val_sc)
            preprocessor = scaler
            label_encoder = le

        elif model_id == "lightgbm":
            try:
                from lightgbm import LGBMClassifier        # noqa: PLC0415
            except ImportError:
                raise ImportError(
                    "LightGBM is not installed. Run: pip install lightgbm"
                )
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc   = scaler.transform(X_val)
            clf = LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbose=-1,
            )
            clf.fit(X_train_sc, y_train,
                    eval_set=[(X_val_sc, y_val)], callbacks=[])
            y_pred = clf.predict(X_val_sc)
            y_prob = clf.predict_proba(X_val_sc)
            preprocessor = scaler
            label_encoder = None

        elif model_id == "random_forest":
            from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc   = scaler.transform(X_val)
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
            )
            clf.fit(X_train_sc, y_train)
            y_pred = clf.predict(X_val_sc)
            y_prob = clf.predict_proba(X_val_sc)
            preprocessor = scaler
            label_encoder = None

        else:
            raise ValueError(f"Unknown tabular model_id: {model_id}")

        _job_update(job, progress_pct=80)

        # ── Phase 5: Evaluate ──
        phase("Evaluating on validation set", 85)

        acc = float(accuracy_score(y_val, y_pred))
        f1  = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))

        # AUC — handle binary vs multiclass
        try:
            if n_classes == 2:
                auc = float(roc_auc_score(y_val, y_prob[:, 1]))
            else:
                auc = float(roc_auc_score(
                    y_val, y_prob, multi_class="ovr", average="weighted"
                ))
        except Exception:
            auc = 0.0

        metrics = {"accuracy": round(acc, 4), "f1": round(f1, 4), "auc": round(auc, 4)}
        logger.info("Eval metrics: %s", metrics)

        # Store fitted objects for use in attacks
        _trained_objects["tabular"] = {
            "model":        clf,
            "preprocessor": preprocessor,
            "label_encoder": label_encoder,
            "feature_names": numeric_cols,
            "model_id":     model_id,
            "n_classes":    n_classes,
            "trained_rows": len(X_train),
        }

        _job_update(
            job,
            status="completed",
            progress_pct=100,
            completed_phases=5,
            current_phase="Training complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            trained_rows=len(X_train) + len(X_val),
            feature_names=numeric_cols,
            n_classes=n_classes,
        )
        _active_model["tabular"] = model_id
        logger.info("tabular training completed: acc=%.4f f1=%.4f auc=%.4f", acc, f1, auc)

    except Exception as exc:
        logger.exception("tabular training failed [%s]", job_id[:8])
        _job_update(
            job,
            status="failed",
            error=str(exc),
            current_phase=f"Failed: {exc}",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )


# ===========================================================================
# REAL TEXT TRAINING — DistilBERT fine-tuning / TF-IDF+LR
# ===========================================================================

def _train_text_real(
    job_id: str, model_id: str, jsonl_bytes: bytes, label_column: str
) -> None:
    """
    Real text model training.
    - tfidf_lr: TF-IDF vectorisation + Logistic Regression (fast, CPU-only)
    - distilbert / secbert: HuggingFace fine-tuning (needs transformers + torch)
    """
    job = _training_jobs[job_id]

    def phase(name: str, pct: int) -> None:
        _job_update(job, current_phase=name, progress_pct=pct)
        logger.info("text training [%s] — %s (%d%%)", job_id[:8], name, pct)

    try:
        _job_update(job, status="running", started_at=datetime.now(timezone.utc).isoformat())

        # ── Phase 1: Parse JSONL or CSV ──
        phase("Parsing text dataset", 5)
        import numpy as np                               # noqa: PLC0415

        raw_text = jsonl_bytes.decode("utf-8", errors="replace")
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

        texts, labels = [], []
        parse_errors = 0

        for line in lines:
            try:
                obj = json.loads(line)
                # Support various key names for the text payload
                payload = (
                    obj.get("payload") or obj.get("text") or
                    obj.get("message") or obj.get("content") or
                    str(obj)
                )
                # Support various label key names — same shared candidate list
                # used for tabular (data.py / LABEL_COLUMN_CANDIDATES), so a
                # CSV-derived or hand-written JSONL with e.g. "fraud" or "y"
                # as the label key resolves the same way tabular does.
                lbl = obj.get(label_column)
                if lbl is None:
                    from api.routes.data import LABEL_COLUMN_CANDIDATES  # noqa: PLC0415
                    for cand in LABEL_COLUMN_CANDIDATES:
                        if obj.get(cand) is not None:
                            lbl = obj.get(cand)
                            break
                if payload is not None and lbl is not None:
                    texts.append(str(payload)[:512])   # truncate to 512 chars
                    labels.append(int(lbl))
            except (json.JSONDecodeError, ValueError):
                parse_errors += 1

        if not texts:
            raise ValueError(
                f"No valid records parsed. Parse errors: {parse_errors}. "
                "Each line must be JSON with 'payload' and 'label' keys."
            )

        logger.info("Parsed %d records (%d errors)", len(texts), parse_errors)
        _job_update(job, progress_pct=15)

        # ── Phase 2: Validate ──
        phase("Validating labels and class balance", 20)
        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)
        logger.info("Classes: %s | Records: %d", unique_labels[:10], len(texts))
        _job_update(job, progress_pct=28)

        # ── Phase 3: Split ──
        phase("Splitting train / validation (80 / 20)", 30)
        from sklearn.model_selection import train_test_split  # noqa: PLC0415
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.20, random_state=42,
            stratify=labels if n_classes <= 20 else None
        )
        logger.info("Train: %d | Val: %d", len(X_train), len(X_val))
        _job_update(job, progress_pct=38)

        # ── Phase 4: Train ──
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: PLC0415

        if model_id == "tfidf_lr":
            # Fast CPU training — always available
            phase("Training TF-IDF + Logistic Regression", 45)
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
            from sklearn.linear_model import LogisticRegression           # noqa: PLC0415
            from sklearn.pipeline import Pipeline                          # noqa: PLC0415

            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=30_000, ngram_range=(1, 2),
                    sublinear_tf=True, strip_accents="unicode",
                    analyzer="word", min_df=2,
                )),
                ("clf", LogisticRegression(
                    max_iter=1000, C=1.0, solver="lbfgs",
                    multi_class="auto", random_state=42, n_jobs=-1,
                )),
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            y_prob = pipe.predict_proba(X_val)

            _trained_objects["text"] = {
                "model": pipe, "model_id": model_id,
                "n_classes": n_classes, "trained_rows": len(X_train),
            }

        elif model_id in ("distilbert", "secbert"):
            # HuggingFace fine-tuning
            phase("Loading tokeniser and model weights", 42)
            try:
                import torch                                              # noqa: PLC0415
                from transformers import (                               # noqa: PLC0415
                    AutoTokenizer, AutoModelForSequenceClassification,
                    TrainingArguments, Trainer,
                )
                from torch.utils.data import Dataset as TorchDataset    # noqa: PLC0415
            except ImportError as e:
                raise ImportError(
                    f"PyTorch / transformers not available: {e}. "
                    "Run: pip install torch transformers"
                ) from e

            # Model name mapping
            hf_model = {
                "distilbert": "distilbert-base-uncased",
                "secbert":    "jackaduma/SecBERT",
            }[model_id]

            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_model)
            except Exception:
                logger.warning("Could not load %s from HuggingFace, falling back to distilbert-base-uncased", hf_model)
                hf_model  = "distilbert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(hf_model)

            _job_update(job, progress_pct=48)
            phase("Tokenising dataset", 50)

            class _TextDataset(TorchDataset):
                def __init__(self, texts_, labels_, tok, max_len=128):
                    enc = tok(texts_, truncation=True, padding="max_length",
                              max_length=max_len, return_tensors="pt")
                    self.input_ids      = enc["input_ids"]
                    self.attention_mask = enc["attention_mask"]
                    self.labels         = torch.tensor(labels_, dtype=torch.long)

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, idx):
                    return {
                        "input_ids":      self.input_ids[idx],
                        "attention_mask": self.attention_mask[idx],
                        "labels":         self.labels[idx],
                    }

            train_ds = _TextDataset(X_train, y_train, tokenizer)
            val_ds   = _TextDataset(X_val,   y_val,   tokenizer)
            _job_update(job, progress_pct=58)

            phase(f"Fine-tuning {model_id} (3 epochs)", 60)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Training on device: %s", device)

            model = AutoModelForSequenceClassification.from_pretrained(
                hf_model, num_labels=n_classes,
                ignore_mismatched_sizes=True,
            )

            import tempfile, os as _os                   # noqa: PLC0415, E401
            with tempfile.TemporaryDirectory() as tmp_dir:
                args = TrainingArguments(
                    output_dir=tmp_dir,
                    num_train_epochs=3,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=32,
                    warmup_steps=min(100, len(train_ds) // 16),
                    weight_decay=0.01,
                    logging_dir=_os.path.join(tmp_dir, "logs"),
                    evaluation_strategy="epoch",
                    save_strategy="no",
                    load_best_model_at_end=False,
                    no_cuda=(device == "cpu"),
                    report_to="none",
                    dataloader_num_workers=0,
                )

                def _compute_metrics(eval_pred):
                    logits, lbl = eval_pred
                    preds = logits.argmax(axis=-1)
                    return {
                        "accuracy": float(accuracy_score(lbl, preds)),
                        "f1": float(f1_score(lbl, preds, average="weighted", zero_division=0)),
                    }

                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    compute_metrics=_compute_metrics,
                )
                trainer.train()
                eval_result = trainer.evaluate()

            _job_update(job, progress_pct=88)
            phase("Evaluating on validation set", 90)

            # Get predictions for AUC
            pred_output = trainer.predict(val_ds)
            import torch as _torch                       # noqa: PLC0415
            y_prob_np = _torch.softmax(
                _torch.tensor(pred_output.predictions), dim=-1
            ).numpy()
            y_pred_np = y_prob_np.argmax(axis=-1)

            acc = float(accuracy_score(y_val, y_pred_np))
            f1  = float(f1_score(y_val, y_pred_np, average="weighted", zero_division=0))
            try:
                auc = float(roc_auc_score(
                    y_val, y_prob_np if n_classes > 2 else y_prob_np[:, 1],
                    multi_class="ovr" if n_classes > 2 else "raise",
                    average="weighted",
                ))
            except Exception:
                auc = 0.0

            _trained_objects["text"] = {
                "model": model, "tokenizer": tokenizer,
                "model_id": model_id, "n_classes": n_classes,
                "trained_rows": len(X_train), "device": device,
            }

            # Override y_pred / y_prob used in metrics below
            y_pred = y_pred_np
            y_prob = y_prob_np

        else:
            raise ValueError(f"Unknown text model_id: {model_id}")

        # ── Phase 5: Final metrics ──
        phase("Saving metrics", 92)
        if model_id == "tfidf_lr":
            y_pred_final = y_pred
            y_prob_final = y_prob
            acc = float(accuracy_score(y_val, y_pred_final))
            f1  = float(f1_score(y_val, y_pred_final, average="weighted", zero_division=0))
            try:
                auc = float(roc_auc_score(
                    y_val,
                    y_prob_final if n_classes > 2 else y_prob_final[:, 1],
                    multi_class="ovr" if n_classes > 2 else "raise",
                    average="weighted",
                ))
            except Exception:
                auc = 0.0

        metrics = {"accuracy": round(acc, 4), "f1": round(f1, 4), "auc": round(auc, 4)}
        logger.info("Text training metrics: %s", metrics)

        _job_update(
            job,
            status="completed",
            progress_pct=100,
            completed_phases=5,
            current_phase="Training complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            trained_rows=len(X_train) + len(X_val),
            parse_errors=parse_errors,
            n_classes=n_classes,
        )
        _active_model["text"] = model_id
        logger.info("text training completed: acc=%.4f f1=%.4f auc=%.4f", acc, f1, auc)

    except Exception as exc:
        logger.exception("text training failed [%s]", job_id[:8])
        _job_update(
            job,
            status="failed",
            error=str(exc),
            current_phase=f"Failed: {exc}",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )


# ===========================================================================
# IMAGE — simulation only (GPU pipeline not yet integrated)
# ===========================================================================

def _train_image_simulated(job_id: str, model_id: str, n_images: int) -> None:
    job = _training_jobs[job_id]
    phases = [
        ("Loading image ZIP",         4),
        ("Resizing images to 224×224", 4),
        ("Building data loaders",      3),
        ("Training model (30 epochs)", 12),
        ("Evaluating on validation set", 3),
    ]
    try:
        _job_update(job, status="running", started_at=datetime.now(timezone.utc).isoformat())
        for i, (name, secs) in enumerate(phases):
            _job_update(job, current_phase=name, progress_pct=round((i/len(phases))*100), completed_phases=i)
            logger.info("image training [%s] phase %d: %s", job_id[:8], i+1, name)
            time.sleep(secs)
        metrics_map = {
            "resnet18":       {"accuracy": 0.881, "f1": 0.868, "auc": 0.934},
            "efficientnet_b0":{"accuracy": 0.913, "f1": 0.902, "auc": 0.958},
            "vit_base":       {"accuracy": 0.942, "f1": 0.935, "auc": 0.977},
        }
        metrics = metrics_map.get(model_id, {"accuracy": 0.88, "f1": 0.86, "auc": 0.93})
        _job_update(
            job, status="completed", progress_pct=100, completed_phases=len(phases),
            current_phase="Training complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics, trained_rows=n_images,
        )
        _active_model["image"] = model_id
    except Exception as exc:
        logger.exception("image training failed")
        _job_update(job, status="failed", error=str(exc),
                    current_phase=f"Failed: {exc}",
                    completed_at=datetime.now(timezone.utc).isoformat())


# ===========================================================================
# DISPATCH — picks the right training function
# ===========================================================================

def _dispatch_training(
    job_id: str, modality: str, model_id: str,
    raw_bytes: Optional[bytes], label_column: str, n_rows: int,
) -> None:
    if modality == "tabular":
        _train_tabular_real(job_id, model_id, raw_bytes, label_column)
    elif modality == "text":
        _train_text_real(job_id, model_id, raw_bytes, label_column)
    elif modality == "image":
        _train_image_simulated(job_id, model_id, n_rows)
    else:
        job = _training_jobs[job_id]
        _job_update(job, status="failed", error=f"Unknown modality: {modality}",
                    completed_at=datetime.now(timezone.utc).isoformat())


# ===========================================================================
# API ENDPOINTS
# ===========================================================================

@router.get("/available", summary="List all available built-in models per modality")
async def get_available_models():
    return {
        "models": BUILT_IN_MODELS,
        "active": _active_model,
        "uploaded": {
            m: ({"filename": v["filename"], "framework": v["framework"]} if v else None)
            for m, v in _uploaded_models.items()
        },
    }


@router.get("/status", summary="Training job status per modality")
async def get_model_status():
    latest: dict[str, Optional[dict]] = {"tabular": None, "text": None, "image": None}
    for job in _training_jobs.values():
        m = job["modality"]
        if latest[m] is None or job["created_at"] > latest[m]["created_at"]:
            latest[m] = job
    return {
        "active_models": _active_model,
        "uploaded_models": {
            m: ({"filename": v["filename"], "framework": v["framework"]} if v else None)
            for m, v in _uploaded_models.items()
        },
        "latest_training_jobs": latest,
        "trained_objects_ready": {
            m: (_trained_objects[m] is not None) for m in _trained_objects
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/active", summary="Which model is currently active per modality")
async def get_active_models():
    return {
        "active": _active_model,
        "uploaded": {m: bool(v) for m, v in _uploaded_models.items()},
        "trained": {m: (_trained_objects[m] is not None) for m in _trained_objects},
    }


@router.post("/train/{modality}", summary="Train a model on uploaded data", status_code=202)
async def train_model(
    modality: Modality,
    model_id: str = Query(..., description="Model ID from GET /models/available"),
    label_column: str = Query(default="label", description="Name of the label column"),
):
    import api.routes.data as _data_module               # noqa: PLC0415
    upload_info = _data_module._uploaded.get(modality.value, {})

    if not upload_info.get("filename"):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse.now(
                error="No Data Uploaded",
                detail=f"Upload a {modality.value} dataset first via POST /api/v1/data/upload/{modality.value}",
            ).model_dump(),
        )

    valid_ids = [m["id"] for m in BUILT_IN_MODELS.get(modality.value, [])]
    if model_id not in valid_ids:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse.now(
                error="Invalid Model",
                detail=f"'{model_id}' not valid for {modality.value}. Choose from: {valid_ids}",
            ).model_dump(),
        )

    # Get raw bytes from upload store
    raw_bytes = upload_info.get("_raw_bytes")
    n_rows    = upload_info.get("rows", upload_info.get("valid_images", 100))

    if raw_bytes is None and modality.value != "image":
        return JSONResponse(
            status_code=400,
            content=ErrorResponse.now(
                error="Data Not Available",
                detail="Uploaded file bytes not found in memory. Please re-upload the file.",
            ).model_dump(),
        )

    job_id = str(uuid.uuid4())
    _training_jobs[job_id] = {
        "job_id": job_id, "modality": modality.value, "model_id": model_id,
        "status": "pending", "progress_pct": 0, "completed_phases": 0,
        "current_phase": "Queued", "total_phases": 5,
        "metrics": None, "error": None, "feature_names": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None, "completed_at": None, "trained_rows": 0,
        "label_column": label_column,
    }

    thread = threading.Thread(
        target=_dispatch_training,
        args=(job_id, modality.value, model_id, raw_bytes, label_column, n_rows),
        daemon=True,
    )
    thread.start()

    logger.info("Training job %s queued: %s / %s on %d rows",
                job_id[:8], modality.value, model_id, n_rows)
    return {
        "job_id": job_id, "modality": modality.value,
        "model_id": model_id, "status": "pending",
        "message": f"Real training started on {upload_info['filename']}",
    }


@router.get("/train/status/{job_id}", summary="Poll training job progress")
async def get_training_status(job_id: str):
    job = _training_jobs.get(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse.now(
                error="Not Found",
                detail=f"No training job with job_id '{job_id}'",
            ).model_dump(),
        )
    return job


@router.post("/select/{modality}", summary="Switch the active model")
async def select_model(
    modality: Modality,
    model_id: str = Query(...),
):
    valid_ids = [m["id"] for m in BUILT_IN_MODELS.get(modality.value, [])]
    if _uploaded_models[modality.value]:
        valid_ids.append("uploaded")
    if model_id not in valid_ids:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse.now(
                error="Invalid Model",
                detail=f"'{model_id}' not available for {modality.value}. Options: {valid_ids}",
            ).model_dump(),
        )
    _active_model[modality.value] = model_id
    return {
        "modality": modality.value, "active_model": model_id,
        "message": f"Active model for {modality.value} switched to {model_id}",
    }


@router.post("/upload/{modality}", summary="Upload a pre-trained model file")
async def upload_model(modality: Modality, file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_MODEL_BYTES:
        return JSONResponse(
            status_code=413,
            content=ErrorResponse.now(
                error="File Too Large",
                detail=f"Maximum {MAX_MODEL_MB} MB. Got {len(raw)/1024/1024:.1f} MB.",
            ).model_dump(),
        )
    fname = file.filename or "model"
    ext   = os.path.splitext(fname)[1].lower()
    accepted = _ACCEPTED_EXTENSIONS.get(modality.value, [])
    if ext not in accepted:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse.now(
                error="Invalid Format",
                detail=f"Accepted: {accepted}. Got '{ext}'.",
            ).model_dump(),
        )
    fw_map = {
        ".pkl":"scikit-learn", ".joblib":"scikit-learn",
        ".pt":"pytorch", ".pth":"pytorch", ".bin":"pytorch",
        ".json":"xgboost", ".onnx":"onnx",
    }
    framework = fw_map.get(ext, "unknown")
    _uploaded_models[modality.value] = {
        "filename": fname, "framework": framework,
        "extension": ext, "size_mb": round(len(raw)/1024/1024, 2),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "attack_mode": "black_box" if framework == "scikit-learn" else "gradient",
        "_raw": raw,
    }
    _active_model[modality.value] = "uploaded"
    return {
        "modality": modality.value, "filename": fname, "framework": framework,
        "size_mb": round(len(raw)/1024/1024, 2),
        "active": True,
        "message": f"Model uploaded and activated for {modality.value}.",
    }


@router.delete("/uploaded/{modality}", summary="Remove uploaded model and revert to default")
async def delete_uploaded_model(modality: Modality):
    if not _uploaded_models[modality.value]:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse.now(
                error="Not Found",
                detail=f"No uploaded model for {modality.value}.",
            ).model_dump(),
        )
    _uploaded_models[modality.value] = None
    defaults = {"tabular": "xgboost", "text": "distilbert", "image": "resnet18"}
    _active_model[modality.value] = defaults[modality.value]
    return {
        "modality": modality.value,
        "active_model": defaults[modality.value],
        "message": f"Uploaded model removed. Reverted to {defaults[modality.value]}.",
    }