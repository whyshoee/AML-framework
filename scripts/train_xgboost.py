"""
[Partner B]
train_xgboost.py — Trains an XGBoost binary classifier on tabular honeypot /
synthetic network-traffic data for the AML FinTech API Security Framework.

Pipeline:
    1. Load synthetic_traffic.csv (or real merged data once Phase 2 finishes).
    2. One-hot encode booleans, split 70/15/15 (stratified).
    3. Fit a StandardScaler preprocessing pipeline, persist it.
    4. Hyperparameter-search a small grid with 3-fold CV.
    5. Refit the best model with early stopping against the validation set.
    6. Evaluate on the held-out test set (accuracy/precision/recall/F1/AUC).
    7. Save model, metrics, and diagnostic plots.
    8. Expose predict_proba_from_raw() for downstream use (Phase 6 API / attacks).

Run:
    python scripts/train_xgboost.py
    python scripts/train_xgboost.py --data-path data/processed/tabular/cleaned_final.csv
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train_xgboost")

# ---------------------------------------------------------------------------
# Feature configuration. Kept at module level so predict_proba_from_raw() and
# any downstream importer (Phase 4 attacks, Phase 6 API) share a single
# source of truth for column order and dtypes.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "src_port",
    "dst_port",
    "bytes_received",
    "connection_duration_ms",
    "payload_entropy",
    "packet_size",
]
BOOLEAN_FEATURES = ["is_repeated_src"]
TARGET_COL = "label"

# Final column order fed to the model after one-hot encoding the boolean.
# is_repeated_src becomes is_repeated_src_True / is_repeated_src_False via
# pd.get_dummies; we fix the order here so train/val/test and any later
# inference call always produce identically-shaped feature vectors.
FEATURE_COLUMNS = NUMERIC_FEATURES + ["is_repeated_src_True", "is_repeated_src_False"]


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------
def load_and_prepare_data(data_path: str):
    """Load the CSV, one-hot encode booleans, and return X (DataFrame), y (Series)."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run scripts/generate_synthetic_tabular.py first, "
            "or pass --data-path pointing at the merged/cleaned dataset."
        )

    df = pd.read_csv(path)
    logger.info("Loaded %s rows from %s", len(df), path)

    missing = set(NUMERIC_FEATURES + BOOLEAN_FEATURES + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # One-hot encode boolean feature(s). get_dummies on a bool column yields
    # True/False columns; we reindex to FEATURE_COLUMNS so any column absent
    # in this particular slice (e.g. all-True in a tiny sample) is filled
    # with 0 rather than silently dropped.
    encoded = pd.get_dummies(df[BOOLEAN_FEATURES[0]], prefix=BOOLEAN_FEATURES[0])
    X = pd.concat([df[NUMERIC_FEATURES], encoded], axis=1)
    X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    y = df[TARGET_COL].astype(int)

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """70/15/15 stratified train/val/test split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    # 0.5 of the remaining 30% -> 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def build_and_fit_preprocessor(X_train: pd.DataFrame, output_dir: Path) -> Pipeline:
    """Fit a StandardScaler-only sklearn Pipeline on the training features and persist it."""
    pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    pipeline.fit(X_train)

    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = output_dir / "preprocessor.pkl"
    joblib.dump(pipeline, preprocessor_path)
    logger.info("Saved fitted preprocessing pipeline to %s", preprocessor_path)
    return pipeline


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def hyperparameter_search(X_train_scaled, y_train, seed: int = 42) -> dict:
    """
    Small grid search over max_depth / learning_rate with 3-fold CV.

    NOTE: this search uses a fixed, modest n_estimators (100) purely to keep the
    grid search itself fast. The *final* model below is refit from scratch with
    n_estimators=300 and early stopping against the validation set, using
    whatever {max_depth, learning_rate} this search finds best.
    """
    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
    }

    base_estimator = xgb.XGBClassifier(
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )

    search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train_scaled, y_train)

    logger.info("Best hyperparameters from grid search: %s", search.best_params_)
    logger.info("Best CV F1 score: %.4f", search.best_score_)
    return search.best_params_


def train_final_model(
    X_train_scaled, y_train, X_val_scaled, y_val, best_params: dict, seed: int = 42
) -> xgb.XGBClassifier:
    """Refit the final model at full n_estimators with early stopping on the val set."""
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    logger.info(
        "Class balance in train -> negatives: %d, positives: %d, scale_pos_weight: %.3f",
        neg, pos, scale_pos_weight,
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=best_params.get("max_depth", 6),
        learning_rate=best_params.get("learning_rate", 0.05),
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=20,  # modern xgboost: must be set in constructor, not .fit()
        random_state=seed,
        n_jobs=-1,
    )
    # NOTE on API drift: the original spec used use_label_encoder=False
    # (removed/invalid in xgboost>=2.x — the label encoder it guarded against
    # was already removed) and passed early_stopping_rounds to .fit() (moved
    # to the constructor as of xgboost 2.x). Both are corrected here so this
    # script runs on current xgboost rather than erroring at runtime.
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False,
    )
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        logger.info("Early stopping selected best_iteration=%d (of 300)", best_iter)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model: xgb.XGBClassifier, X_test_scaled, y_test) -> dict:
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test_samples": int(len(y_test)),
    }
    logger.info("Test metrics: %s", {k: v for k, v in metrics.items() if k != "confusion_matrix"})
    return metrics, y_proba


def plot_feature_importance(model: xgb.XGBClassifier, feature_names: list, output_path: Path):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in order]
    top_importances = importances[order]

    plt.figure(figsize=(8, 5))
    plt.barh(top_features[::-1], top_importances[::-1], color="#2563EB")
    plt.xlabel("Feature Importance")
    plt.title("XGBoost Top 10 Feature Importances — Network Traffic Classifier")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved feature importance plot to %s", output_path)


def plot_roc(y_test, y_proba, output_path: Path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="#2563EB", label=f"ROC curve (AUC = {roc_auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost ROC Curve — Network Traffic Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved ROC curve plot to %s", output_path)


# ---------------------------------------------------------------------------
# Inference helper — importable by Phase 4 attacks / Phase 6 FastAPI backend
# ---------------------------------------------------------------------------
def predict_proba_from_raw(
    raw_dict: dict,
    model_path: str = "models/xgboost/model.json",
    preprocessor_path: str = "models/xgboost/preprocessor.pkl",
) -> np.ndarray:
    """
    Run inference on a single raw feature dict (e.g. straight from a
    HoneypotEvent payload) and return class probabilities [P(benign), P(attack)].

    raw_dict must contain: src_port, dst_port, bytes_received,
    connection_duration_ms, payload_entropy, packet_size, is_repeated_src (bool).
    """
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    preprocessor: Pipeline = joblib.load(preprocessor_path)

    row = {feat: raw_dict.get(feat, 0) for feat in NUMERIC_FEATURES}
    is_repeated = bool(raw_dict.get("is_repeated_src", False))
    row["is_repeated_src_True"] = 1 if is_repeated else 0
    row["is_repeated_src_False"] = 0 if is_repeated else 1

    X = pd.DataFrame([row]).reindex(columns=FEATURE_COLUMNS, fill_value=0)
    X_scaled = preprocessor.transform(X)
    return model.predict_proba(X_scaled)[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on tabular honeypot data.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/tabular/synthetic_traffic.csv",
        help="Path to the input CSV (synthetic or merged/cleaned real+synthetic).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/xgboost",
        help="Directory to save model, preprocessor, metrics, and plots.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_and_prepare_data(args.data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=args.seed)

    preprocessor = build_and_fit_preprocessor(X_train, output_dir)
    X_train_scaled = preprocessor.transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    best_params = hyperparameter_search(X_train_scaled, y_train, seed=args.seed)
    model = train_final_model(
        X_train_scaled, y_train, X_val_scaled, y_val, best_params, seed=args.seed
    )

    metrics, y_proba = evaluate_model(model, X_test_scaled, y_test)
    metrics["best_hyperparameters"] = best_params
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", output_dir / "training_metrics.json")

    plot_feature_importance(model, FEATURE_COLUMNS, output_dir / "feature_importance.png")
    plot_roc(y_test, y_proba, output_dir / "roc_curve.png")

    model_path = output_dir / "model.json"
    model.save_model(str(model_path))
    logger.info("Saved trained model to %s", model_path)

    logger.info("✅ XGBoost training complete. Test accuracy: %.4f", metrics["accuracy"])


if __name__ == "__main__":
    main()