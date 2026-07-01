# =============================================================================
# attacks/attack_runner.py
# [Partner B]  — Unified Attack Runner
#
# Single entry-point for running any attack on any modality.
# The runner:
#   1. Reads configs/config.yaml to initialise all attack objects.
#   2. Holds references to all three modality adapters.
#   3. Exposes run_all_attacks() and run_single_attack() so calling code
#      only needs to say "run FGSM on tabular data" without knowing which
#      adapter or AttackConfig to build.
#   4. Persists results to SQLite (via sqlmodel) and to disk.
#
# Designed to be imported by:
#   - scripts/run_framework.py  (Phase 6 orchestrator)
#   - api/routes/attacks.py     (FastAPI routes)
#   - tests/test_integration.py (integration tests)
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

# ── Attack implementations (Partner A) ───────────────────────────────────────
from attacks.base import AttackConfig, AttackResult
from attacks.fgsm     import FGSM
from attacks.pgd      import PGD
from attacks.cw       import CarliniWagner
from attacks.deepfool import DeepFool

# ── Modality adapters (Partner B) ────────────────────────────────────────────
from attacks.adapters.image_adapter   import ImageAdapter
from attacks.adapters.text_adapter    import TextAdapter
from attacks.adapters.tabular_adapter import TabularAdapter

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# Valid modality strings — used for validation throughout
VALID_MODALITIES = ("image", "text", "tabular")
VALID_ATTACKS    = ("FGSM", "PGD", "CW", "DeepFool")


# =============================================================================
# Helper — build AttackConfig from a raw YAML dict for one attack section
# =============================================================================
def _build_config(attack_cfg: dict, global_clip: tuple = (0.0, 1.0)) -> AttackConfig:
    """
    Translate a YAML attack sub-dict into an AttackConfig dataclass.

    Only fields recognised by AttackConfig are copied; any extra keys
    (n_steps, step_size, etc.) are silently ignored here and passed
    separately to the attack constructors.
    """
    return AttackConfig(
        epsilon=float(attack_cfg.get("epsilon", 0.1)),
        clip_min=global_clip[0],
        clip_max=global_clip[1],
    )


# =============================================================================
# AttackRunner — the main class
# =============================================================================
class AttackRunner:
    """
    Orchestrates all adversarial attacks across all modalities.

    Usage
    -----
    >>> runner = AttackRunner("configs/config.yaml")
    >>> results = runner.run_all_attacks("image", model, images, labels)
    >>> print(results["FGSM"].adversarial_accuracy)
    """

    # ------------------------------------------------------------------
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Load the YAML configuration and initialise all attack objects and
        modality adapters.  Nothing heavy is loaded here — model loading is
        deferred to the caller.

        Args:
            config_path: Path to configs/config.yaml (relative or absolute).
        """
        self.config_path = config_path
        self.cfg = self._load_config(config_path)

        logger.info("AttackRunner initialising from '%s'", config_path)

        # ── Instantiate the four attacks ─────────────────────────────────────
        self.attacks: Dict[str, Any] = self._init_attacks()

        # ── Instantiate the three modality adapters ───────────────────────────
        self.image_adapter   = ImageAdapter()
        self.text_adapter    = TextAdapter()
        self.tabular_adapter = TabularAdapter()

        logger.info(
            "AttackRunner ready  |  attacks: %s  |  adapters: image, text, tabular",
            list(self.attacks.keys()),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """
        Read and parse configs/config.yaml.
        Falls back to an empty dict if the file is missing (useful in tests
        where the config may not yet exist).
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(
                "Config not found at '%s'. Using built-in defaults.", config_path
            )
            return {}
        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        logger.info("Config loaded from '%s'", config_path)
        return cfg

    def _init_attacks(self) -> Dict[str, Any]:
        """
        Build one instance of each attack class using hyper-parameters from
        config.yaml.  Missing keys fall back to the AttackConfig defaults.
        """
        attack_cfgs = self.cfg.get("attacks", {})

        # ── FGSM ─────────────────────────────────────────────────────────────
        fgsm_cfg = attack_cfgs.get("fgsm", {})
        fgsm = FGSM(config=_build_config(fgsm_cfg))

        # ── PGD ──────────────────────────────────────────────────────────────
        pgd_cfg = attack_cfgs.get("pgd", {})
        pgd = PGD(
            config=_build_config(pgd_cfg),
            n_steps=int(pgd_cfg.get("n_steps", 40)),
            step_size=float(pgd_cfg.get("step_size", 0.01)),
            random_start=bool(pgd_cfg.get("random_start", True)),
        )

        # ── Carlini–Wagner ───────────────────────────────────────────────────
        cw_cfg = attack_cfgs.get("cw", {})
        cw = CarliniWagner(
            config=_build_config(cw_cfg),
            c=float(cw_cfg.get("c", 1e-4)),
            kappa=float(cw_cfg.get("kappa", 0.0)),
            n_steps=int(cw_cfg.get("n_steps", 1000)),
            lr=float(cw_cfg.get("lr", 0.01)),
        )

        # ── DeepFool ─────────────────────────────────────────────────────────
        df_cfg = attack_cfgs.get("deepfool", {})
        deepfool = DeepFool(
            config=_build_config(df_cfg),
            max_iter=int(df_cfg.get("max_iter", 50)),
            overshoot=float(df_cfg.get("overshoot", 0.02)),
        )

        return {"FGSM": fgsm, "PGD": pgd, "CW": cw, "DeepFool": deepfool}

    # ── Public API ────────────────────────────────────────────────────────────

    def run_all_attacks(
        self,
        modality: str,
        model: nn.Module,
        data: Any,
        labels: Any,
        **kwargs,
    ) -> Dict[str, AttackResult]:
        """
        Run all four attacks (FGSM, PGD, CW, DeepFool) on the given modality.

        Args:
            modality: One of "image" | "text" | "tabular".
            model:    The target classifier for this modality.
            data:     Input data — Tensor for image/tabular, List[str] for text.
            labels:   Ground-truth labels — Tensor for image/tabular.
            **kwargs: Extra keyword arguments forwarded to the adapter
                      (e.g. tokenizer= for the text adapter).

        Returns:
            Dict mapping attack name → AttackResult.

        Raises:
            ValueError: If `modality` is not one of the valid options.
        """
        self._validate_modality(modality)

        results: Dict[str, AttackResult] = {}
        for attack_name, attack_obj in self.attacks.items():
            logger.info("Running %s on modality='%s'", attack_name, modality)
            try:
                result = self._dispatch(
                    attack_name, attack_obj, modality, model, data, labels, **kwargs
                )
                result.attack_name = attack_name
                result.modality    = modality
                results[attack_name] = result
                logger.info(
                    "  ✓ %s  |  ASR=%.2f%%  |  AdvAcc=%.2f%%  |  norm=%.4f",
                    attack_name,
                    result.attack_success_rate * 100,
                    result.adversarial_accuracy * 100,
                    result.perturbation_norm,
                )
            except Exception as exc:
                logger.error("  ✗ %s failed: %s", attack_name, exc)
                # Store a sentinel result so callers always get 4 keys
                results[attack_name] = self._empty_result(attack_name, modality)

        return results

    # ------------------------------------------------------------------
    def run_single_attack(
        self,
        attack_name: str,
        modality: str,
        model: nn.Module,
        data: Any,
        labels: Any,
        **kwargs,
    ) -> AttackResult:
        """
        Run one named attack on one modality.

        Args:
            attack_name: "FGSM" | "PGD" | "CW" | "DeepFool"
            modality:    "image" | "text" | "tabular"
            model:       Target classifier.
            data:        Input data.
            labels:      Ground-truth labels.

        Returns:
            AttackResult for the single attack.

        Raises:
            ValueError: If attack_name or modality is unrecognised.
        """
        self._validate_modality(modality)
        self._validate_attack_name(attack_name)

        attack_obj = self.attacks[attack_name]
        logger.info(
            "run_single_attack: %s on modality='%s'", attack_name, modality
        )
        result = self._dispatch(
            attack_name, attack_obj, modality, model, data, labels, **kwargs
        )
        result.attack_name = attack_name
        result.modality    = modality
        return result

    # ------------------------------------------------------------------
    def save_results(
        self,
        results: Dict[str, AttackResult],
        output_dir: str,
        db_session=None,
    ) -> None:
        """
        Persist attack results to disk and optionally to the SQLite database.

        Disk layout:
          output_dir/
            {modality}/
              {attack_name}_x_adv.pt        # adversarial tensor
              {attack_name}_metadata.json   # human-readable metrics

        Args:
            results:    Dict returned by run_all_attacks() or run_single_attack().
            output_dir: Root directory under data/adversarial/.
            db_session: An active SQLModel Session.  If None, DB write is skipped.
        """
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        for attack_name, result in results.items():
            modality = getattr(result, "modality", "unknown")
            mod_dir  = out_root / modality
            mod_dir.mkdir(parents=True, exist_ok=True)

            # ── Save adversarial tensor / text to disk ──────────────────────
            safe_name = attack_name.replace(" ", "_").lower()

            if isinstance(result.x_adv, torch.Tensor):
                tensor_path = mod_dir / f"{safe_name}_x_adv.pt"
                torch.save(result.x_adv.cpu(), tensor_path)
                logger.info("Saved adversarial tensor → %s", tensor_path)
            elif isinstance(result.x_adv, list):
                # Text adapter returns a list of dicts
                text_path = mod_dir / f"{safe_name}_perturbed_texts.json"
                with open(text_path, "w") as fh:
                    json.dump(result.x_adv, fh, indent=2)
                logger.info("Saved perturbed texts → %s", text_path)

            # ── Save human-readable metadata ────────────────────────────────
            meta = {
                "attack_name":        attack_name,
                "modality":           modality,
                "epsilon":            result.epsilon,
                "perturbation_norm":  result.perturbation_norm,
                "attack_success_rate": result.attack_success_rate,
                "adversarial_accuracy": result.adversarial_accuracy,
                "clean_accuracy":     result.clean_accuracy,
                "success":            result.success,
                "iterations":         result.iterations,
                "saved_at":           datetime.utcnow().isoformat(),
            }
            meta_path = mod_dir / f"{safe_name}_metadata.json"
            with open(meta_path, "w") as fh:
                json.dump(meta, fh, indent=2)
            logger.info("Saved metadata → %s", meta_path)

            # ── Persist to SQLite if a session was provided ──────────────────
            if db_session is not None:
                self._save_to_db(result, meta, db_session)

    # ── Internal routing ──────────────────────────────────────────────────────

    def _dispatch(
        self,
        attack_name: str,
        attack_obj: Any,
        modality: str,
        model: nn.Module,
        data: Any,
        labels: Any,
        **kwargs,
    ) -> AttackResult:
        """
        Route data + model to the correct adapter method.

        Image   → ImageAdapter.apply_attack()
        Text    → TextAdapter.apply_attack()  (requires tokenizer= in kwargs)
        Tabular → TabularAdapter.apply_attack()
        """
        if modality == "image":
            # data must be a float Tensor [B, C, H, W] in normalised space
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            return self.image_adapter.apply_attack(attack_obj, model, data, labels)

        elif modality == "text":
            # data must be List[str]; a tokenizer kwarg is expected
            tokenizer = kwargs.pop("tokenizer", None)
            if tokenizer is None:
                raise ValueError(
                    "Text modality requires 'tokenizer' keyword argument."
                )
            text_results = self.text_adapter.apply_attack(
                attack_obj, model, tokenizer, data, **kwargs
            )
            # Wrap the list-of-dicts in an AttackResult for a consistent interface
            return self._text_results_to_attack_result(text_results, attack_name)

        elif modality == "tabular":
            # data must be a numpy array [N, F]; labels a numpy array [N]
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            feature_names = kwargs.get("feature_names")
            if feature_names is None:
                raise ValueError("Tabular modality requires 'feature_names' keyword argument.")
            feature_bounds = kwargs.get("feature_bounds")
            return self.tabular_adapter.apply_attack(
                attack_obj, model, data, labels, feature_names, feature_bounds
            )

        else:
            raise ValueError(f"Unrecognised modality '{modality}'.")

    @staticmethod
    def _text_results_to_attack_result(
        text_results: List[Dict], attack_name: str
    ) -> AttackResult:
        """
        Convert the List[dict] returned by TextAdapter.apply_attack()
        into an AttackResult so run_all_attacks() can treat all modalities
        uniformly.

        The tensor fields (x_orig, x_adv, y_true, y_pred_*) are stored as
        1-D tensors of success indicators rather than raw embeddings because
        embeddings are not serialisable and are not needed by evaluators.
        """
        n       = len(text_results)
        success = torch.tensor([int(r.get("success", False)) for r in text_results])
        labels  = torch.tensor([r.get("label", 0)           for r in text_results])

        # Build fake "prediction" tensors: 1 = fooled, 0 = not fooled
        # This satisfies the AttackResult interface for metric computation.
        y_pred_orig = labels.clone()                     # correct by definition
        y_pred_adv  = torch.where(success.bool(),
                                   1 - labels,           # flipped
                                   labels)               # not flipped

        asr = success.float().mean().item()

        return AttackResult(
            x_orig=labels,            # placeholder
            x_adv=success,            # placeholder (list stored in extra)
            y_true=labels,
            y_pred_orig=y_pred_orig,
            y_pred_adv=y_pred_adv,
            perturbation_norm=float(
                torch.tensor([r.get("n_tokens_changed", 0)
                               for r in text_results]).mean().item()
            ),
            success=asr > 0.0,
            attack_name=attack_name,
            epsilon=0.0,
            iterations=1,
            modality="text",
            extra={"text_results": text_results},
        )

    @staticmethod
    def _empty_result(attack_name: str, modality: str) -> AttackResult:
        """Sentinel AttackResult for failed attack runs (e.g. OOM or import error)."""
        dummy = torch.zeros(1)
        return AttackResult(
            x_orig=dummy, x_adv=dummy,
            y_true=dummy.long(), y_pred_orig=dummy.long(), y_pred_adv=dummy.long(),
            perturbation_norm=0.0, success=False,
            attack_name=attack_name, epsilon=0.0, iterations=0,
            modality=modality,
            extra={"error": "Attack failed — see logs for details"},
        )

    @staticmethod
    def _save_to_db(result: AttackResult, meta: dict, db_session) -> None:
        """
        Attempt to write an AdversarialSample record to SQLite.
        Import is deferred so the file can be used without the database
        module (e.g. in unit tests with no DB configured).
        """
        try:
            # database.py is at the project root
            root = Path(__file__).resolve().parents[1]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from database import AdversarialSample  # type: ignore

            record = AdversarialSample(
                source_event_id=0,                   # not linked to a honeypot event
                attack_type=meta["attack_name"],
                epsilon=meta["epsilon"],
                modality=meta["modality"],
                perturbed_payload_path=str(
                    meta.get("saved_at", "")
                ),
                original_prediction=str(meta.get("clean_accuracy", "")),
                adversarial_prediction=str(meta.get("adversarial_accuracy", "")),
                attack_success=meta["success"],
                timestamp=datetime.utcnow(),
            )
            db_session.add(record)
            db_session.commit()
            logger.info("DB record saved for %s/%s", meta["modality"], meta["attack_name"])
        except Exception as exc:
            logger.warning("DB write skipped: %s", exc)

    # ── Validation helpers ────────────────────────────────────────────────────

    @staticmethod
    def _validate_modality(modality: str) -> None:
        if modality not in VALID_MODALITIES:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Must be one of {VALID_MODALITIES}."
            )

    @staticmethod
    def _validate_attack_name(attack_name: str) -> None:
        if attack_name not in VALID_ATTACKS:
            raise ValueError(
                f"Unknown attack '{attack_name}'. "
                f"Must be one of {VALID_ATTACKS}."
            )


# =============================================================================
# Demo helper models — used ONLY in the __main__ block below
# These produce random logits so we can verify the runner works end-to-end
# without trained weights.
# =============================================================================

class _DummyImageModel(nn.Module):
    """Tiny CNN that classifies 3×32×32 images into 2 classes."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, 2),
        )

    def forward(self, x):
        return self.net(x)

    # Required by ImageAdapter's _NormWrapper (get_input_embeddings does not apply)
    def get_input_embeddings(self):
        raise AttributeError("Not a text model")


class _DummyTabularModel(nn.Module):
    """Shallow MLP for 8-feature binary classification."""
    def __init__(self, n_features: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


class _DummyTextModel(nn.Module):
    """
    Minimal 'DistilBERT-like' model exposing the attributes TextAdapter needs:
      - get_input_embeddings() → embedding layer
      - forward(inputs_embeds, attention_mask) → object with .logits
    """

    def __init__(self, vocab_size: int = 100, hidden: int = 32, seq_len: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden     = hidden
        self.seq_len    = seq_len

        # Mimic DistilBERT's embedding attribute path
        class _EmbWrapper(nn.Module):
            def __init__(self, emb):
                super().__init__()
                self.word_embeddings = emb

        self._embedding     = nn.Embedding(vocab_size, hidden)
        self.distilbert     = nn.Module()
        self.distilbert.embeddings = _EmbWrapper(self._embedding)
        self._classifier    = nn.Linear(hidden, 2)

    def get_input_embeddings(self):
        return self._embedding

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
            emb = inputs_embeds
        else:
            emb = self._embedding(input_ids)        # [B, L, H]

        # Pool over sequence length → [B, H]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = emb.mean(dim=1)

        logits = self._classifier(pooled)           # [B, 2]

        # Return an object with a .logits attribute (matches HuggingFace API)
        class _Output:
            pass
        out        = _Output()
        out.logits = logits
        return out


# =============================================================================
# __main__ — demonstration run on random dummy data for all three modalities
# =============================================================================

if __name__ == "__main__":
    import sys

    # ── Add project root to sys.path so relative imports work when running
    # this file directly (e.g.  python attacks/attack_runner.py)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("=" * 70)
    print("  AML FinTech Framework — AttackRunner Demo")
    print("  All three modalities, all four attacks, random dummy models")
    print("=" * 70)

    # Locate config (supports running from project root or from attacks/ subdir)
    cfg_candidates = [
        project_root / "configs" / "config.yaml",
        Path("configs") / "config.yaml",
    ]
    cfg_path = next((str(p) for p in cfg_candidates if p.exists()), "configs/config.yaml")
    runner   = AttackRunner(config_path=cfg_path)

    # ────────────────────────────────────────────────────────────────────
    # 1. IMAGE MODALITY — 4 samples of 3×32×32 KYC-proxy images
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("▶  IMAGE MODALITY (ResNet-18 proxy — random weights)")
    print("─" * 70)

    image_model  = _DummyImageModel()
    image_model.eval()

    # Normalised images in [-∞, ∞]; ImageAdapter handles de-norm internally
    images = torch.randn(4, 3, 32, 32)   # [B, C, H, W]
    labels = torch.randint(0, 2, (4,))   # binary labels

    img_results = runner.run_all_attacks("image", image_model, images, labels)

    for name, res in img_results.items():
        print(
            f"  {name:<10}  |  CleanAcc={res.clean_accuracy:.2%}"
            f"  AdvAcc={res.adversarial_accuracy:.2%}"
            f"  ASR={res.attack_success_rate:.2%}"
            f"  ‖δ‖₂={res.perturbation_norm:.4f}"
            f"  iters={res.iterations}"
        )

    # ────────────────────────────────────────────────────────────────────
    # 2. TABULAR MODALITY — 32 samples of 8 network-traffic features
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("▶  TABULAR MODALITY (XGBoost proxy — random weights)")
    print("─" * 70)

    tabular_model = _DummyTabularModel(n_features=8)
    tabular_model.eval()

    # Simulate normalised XGBoost features (already scaled to ~N(0,1))
    X_tab = np.random.randn(32, 8).astype(np.float32)
    y_tab = np.random.randint(0, 2, (32,)).astype(np.int64)

    tab_results = runner.run_all_attacks(
        "tabular",
        tabular_model,
        X_tab,
        y_tab,
        feature_names=["src_port", "dst_port", "bytes_received",
                        "connection_duration_ms", "payload_entropy",
                        "packet_size", "is_repeated_src", "src_ip_frequency"],
    )

    for name, res in tab_results.items():
        print(
            f"  {name:<10}  |  CleanAcc={res.clean_accuracy:.2%}"
            f"  AdvAcc={res.adversarial_accuracy:.2%}"
            f"  ASR={res.attack_success_rate:.2%}"
            f"  ‖δ‖₂={res.perturbation_norm:.4f}"
        )

    # ────────────────────────────────────────────────────────────────────
    # 3. TEXT MODALITY — 4 raw JSON-API payload strings
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("▶  TEXT MODALITY (DistilBERT proxy — random weights)")
    print("─" * 70)

    text_model = _DummyTextModel(vocab_size=200, hidden=32, seq_len=16)
    text_model.eval()

    # Minimal tokeniser shim — maps whitespace-split words to integer IDs
    class _TinyTokenizer:
        """
        Bare-minimum tokeniser compatible with the TextAdapter interface.
        Uses a fixed vocabulary of 200 tokens (word → ID by hash).
        """
        pad_token_id = 0
        vocab_size   = 200

        def __call__(self, text, return_tensors="pt",
                     max_length=16, padding="max_length", truncation=True):
            tokens = text.split()[:max_length]
            ids    = [abs(hash(t)) % 198 + 1 for t in tokens]  # 1..198
            # pad to max_length
            ids   += [0] * (max_length - len(ids))
            ids    = ids[:max_length]
            mask   = [1 if i > 0 else 0 for i in ids]
            return {
                "input_ids":      torch.tensor([ids],  dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long),
            }

        def decode(self, token_ids, skip_special_tokens=True):
            # Convert back to pseudo-words (not semantically meaningful)
            return " ".join(f"tok_{t}" for t in token_ids.tolist() if t > 0)

    tiny_tok = _TinyTokenizer()

    payloads = [
        '{"username": "admin", "password": "secret123"}',
        '{"transaction_id": "TXN001", "amount": 500.0, "currency": "USD"}',
        '{"user_id": "U42", "document_type": "passport", "dob": "1990-01-01"}',
        '{"username": "admin\' OR 1=1--", "password": "ignored"}',
    ]
    text_labels = [0, 0, 0, 1]  # last one is an attack payload

    # Only FGSM and PGD produce meaningful text perturbations with a real model;
    # CW and DeepFool fall back gracefully because the tiny model lacks a full
    # embedding path — this validates the exception-handling in run_all_attacks.
    txt_results = runner.run_all_attacks(
        "text",
        text_model,
        payloads,
        text_labels,            # not a tensor — TextAdapter handles the conversion
        tokenizer=tiny_tok,
    )

    for name, res in txt_results.items():
        extra = res.extra.get("text_results", [])
        n_swapped = sum(r.get("tokens_swapped", 0) for r in extra)
        n_success = sum(int(r.get("success", False)) for r in extra)
        print(
            f"  {name:<10}  |  Samples={len(extra)}"
            f"  Fooled={n_success}/{len(extra)}"
            f"  TotalTokensSwapped={n_swapped}"
        )
        # Print one example perturbed text
        if extra:
            sample = extra[0]
            print(f"             Original : {sample['original_text'][:60]}")
            print(f"             Perturbed: {sample['perturbed_text'][:60]}")

    # ────────────────────────────────────────────────────────────────────
    # 4. Save all results to data/adversarial/ (no DB session here)
    # ────────────────────────────────────────────────────────────────────
    adv_out = str(project_root / "data" / "adversarial")
    print("\n" + "─" * 70)
    print(f"▶  Saving results to {adv_out}/")
    print("─" * 70)

    runner.save_results(img_results,  os.path.join(adv_out, "image"))
    runner.save_results(tab_results,  os.path.join(adv_out, "tabular"))
    runner.save_results(txt_results,  os.path.join(adv_out, "text"))

    print("\n✅  Demo complete.  All results saved.")
    print("    Run  python scripts/run_framework.py --benchmark  for a full sweep.")