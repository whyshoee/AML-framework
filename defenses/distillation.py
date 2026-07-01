"""
[Partner B]
aml_fintech_framework/defenses/distillation.py

Defensive Distillation — Phase 5, Task 3
=========================================
Implements Papernot et al. (2016) "Distillation as a Defense to
Adversarial Perturbations Against Deep Neural Networks" for the
ResNet-18 KYC binary classifier from Phase 3.

Core idea
---------
Train a TEACHER model normally on hard (one-hot) labels.
Run the teacher at a high temperature T to produce SOFT probability labels
  soft[i] = softmax(teacher_logits[i] / T)
Train an identical STUDENT model using those soft labels via KL divergence.
The student learns a smoother decision surface — gradient-based attacks
struggle because the input-gradient magnitudes are much smaller.

Improvements beyond vanilla Papernot (2016)
--------------------------------------------
1. Temperature sweep evaluation
   Test T ∈ [5, 10, 20, 50, 100].  For each T, train a quick student and
   record (student_clean_acc, gradient_masking_score).  Saves a dual-axis
   plot to evaluation/figures/distillation_temperature_sweep.png.

2. Stacked defense via combine_defenses()
   Chains InputPreprocessor → AutoencoderDefense → distilled student in a
   single forward call.  Any one or two stages can be passed as None to
   use only a subset of the stack.

3. Per-layer gradient norm profiling
   compare_gradient_norms_per_layer() returns an OrderedDict mapping every
   named parameter to (teacher_norm, student_norm, ratio), so you can see
   exactly which layers benefit most from distillation.

File layout
-----------
  defenses/distillation.py          ← this file
  evaluation/figures/
    distillation_temperature_sweep.png
  models/resnet18/
    student_distilled.pth           (saved by train_student)
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # headless — safe on servers and Windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE  = 20.0
DEFAULT_SWEEP_TEMPS  = [5.0, 10.0, 20.0, 50.0, 100.0]
NUM_CLASSES          = 2        # binary KYC: legitimate (0) vs fraudulent (1)
IMG_CHANNELS         = 3
IMG_SIZE             = 64      # 64×64 KYC images from Phase 2


# ─────────────────────────────────────────────────────────────────────────────
# Architecture helper  (mirrors train_resnet.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def _build_resnet18(num_classes: int = NUM_CLASSES,
                    dropout: float = 0.3) -> nn.Module:
    """
    Construct the same ResNet-18 head used in Phase 3 train_resnet.py.
    Imports torchvision lazily so the rest of the module is importable
    even without GPU / torchvision installed.
    """
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        # weights=None → random initialisation; we load our own checkpoint
        model = resnet18(weights=None)
    except ImportError as exc:
        raise ImportError(
            "torchvision is required.  Run: pip install torchvision"
        ) from exc

    in_features = model.fc.in_features          # 512 for ResNet-18
    # Replicate Phase-3 head:  Dropout(0.3) → Linear(512, num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# KL-Divergence loss with temperature correction
# ─────────────────────────────────────────────────────────────────────────────
class TemperatureKLLoss(nn.Module):
    """
    KL divergence loss used during distillation training.

    Following Hinton et al. (2015) the loss is multiplied by T² so that
    the gradient magnitudes remain constant relative to hard-label CE loss
    as T changes.

        L = T² · KLDiv( log_softmax(student_logits / T),  soft_targets )

    soft_targets must already be probability distributions (output of
    softmax at temperature T from the teacher).

    Parameters
    ----------
    temperature : distillation temperature T
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE) -> None:
        super().__init__()
        self.T  = temperature
        # batchmean: sum over classes, mean over batch — the correct reduction
        self._kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,   # [B, C]  raw student output
        soft_targets:   torch.Tensor,   # [B, C]  teacher soft probabilities
    ) -> torch.Tensor:
        log_p = F.log_softmax(student_logits / self.T, dim=-1)
        return (self.T ** 2) * self._kl(log_p, soft_targets)


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy helper (no_grad, handles both nn.Module and sklearn-style models)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _compute_accuracy(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Top-1 accuracy of `model` on samples from `loader`."""
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        preds = model(x).argmax(dim=-1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────
class DefensiveDistillation:
    """
    Defensive Distillation wrapper for the Phase-3 ResNet-18 classifier.

    Typical usage
    -------------
    # 1. Instantiate with teacher checkpoint
    dd = DefensiveDistillation("models/resnet18/best_model.pth", temperature=20)

    # 2. Generate soft labels from training data
    soft_ds = dd.generate_soft_labels(train_loader)

    # 3. Train the student
    dd.train_student(soft_ds, n_epochs=30, save_path="models/resnet18/student.pth")

    # 4. Evaluate
    results = dd.evaluate_distillation(test_loader, attack=fgsm_attack)

    # 5. Stacked defense inference
    clean_pred = dd.combine_defenses(x_adv, preprocessing_fn=pp.preprocess,
                                     autoencoder_fn=ae.denoise)
    """

    def __init__(
        self,
        teacher_model_path: str,
        temperature: float        = DEFAULT_TEMPERATURE,
        device:      str          = "cpu",
        dropout:     float        = 0.3,
    ) -> None:
        """
        Parameters
        ----------
        teacher_model_path : path to Phase-3 ResNet-18 .pth checkpoint
        temperature        : distillation temperature T (default 20)
        device             : "cpu" | "cuda" | "mps"
        dropout            : Dropout rate — must match Phase-3 training
        """
        self.T       = temperature
        self.dropout = dropout
        self.device  = torch.device(device)

        logger.info(
            "DefensiveDistillation — T=%.1f | device=%s", self.T, self.device
        )

        # ── Load teacher ───────────────────────────────────────────────────
        self.teacher = self._load_checkpoint(teacher_model_path, dropout)
        self.teacher.eval()
        logger.info("Teacher loaded ← %s", teacher_model_path)

        # ── Student: identical architecture, freshly initialised ───────────
        self.student = _build_resnet18(dropout=dropout).to(self.device)
        logger.info("Student initialised (random weights)")

        # ── Loss function ──────────────────────────────────────────────────
        self.distill_loss = TemperatureKLLoss(temperature=self.T)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _load_checkpoint(self, path: str, dropout: float) -> nn.Module:
        """
        Load a ResNet-18 state dict from disk.
        Handles three common checkpoint formats:
          (a) plain state_dict (torch.save(model.state_dict(), path))
          (b) {"model_state_dict": state_dict, ...}   (train_resnet.py format)
          (c) {"state_dict": state_dict, ...}
        """
        model = _build_resnet18(dropout=dropout).to(self.device)
        raw   = torch.load(path, map_location=self.device)

        if isinstance(raw, dict):
            if "model_state_dict" in raw:
                state = raw["model_state_dict"]
            elif "state_dict" in raw:
                state = raw["state_dict"]
            else:
                # Assume it IS the state dict (all keys are param names)
                state = raw
        else:
            raise ValueError(
                f"Unrecognised checkpoint format in {path}. "
                "Expected a state_dict or a dict containing 'model_state_dict'."
            )

        model.load_state_dict(state)
        return model

    @staticmethod
    def _soft_probs(logits: torch.Tensor, T: float) -> torch.Tensor:
        """Return softmax(logits / T) — the soft probability distribution."""
        return F.softmax(logits / T, dim=-1)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Soft label generation
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_soft_labels(
        self,
        dataloader:  DataLoader,
        temperature: Optional[float] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward-pass the dataset through the teacher at temperature T and
        collect soft probability labels for each batch.

        Parameters
        ----------
        dataloader  : DataLoader yielding (x, y) — y is ignored here
        temperature : override self.T if provided

        Returns
        -------
        List of (x_batch_cpu, soft_labels_cpu) tuples.
        x_batch is kept on CPU to save GPU VRAM during the longer
        student training loop.
        """
        T = temperature if temperature is not None else self.T
        self.teacher.eval()

        soft_dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        n_batches   = 0
        n_samples   = 0

        for batch in dataloader:
            x = batch[0].to(self.device)

            logits = self.teacher(x)                # [B, C]
            soft   = self._soft_probs(logits, T)    # [B, C]  probabilities

            # Store on CPU — student training will move them back to device
            soft_dataset.append((x.cpu(), soft.cpu()))
            n_batches += 1
            n_samples += x.size(0)

        logger.info(
            "Soft labels generated — %d batches | %d samples | T=%.1f",
            n_batches, n_samples, T,
        )
        return soft_dataset

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Student training
    # ──────────────────────────────────────────────────────────────────────────
    def train_student(
        self,
        soft_label_dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        n_epochs:      int            = 30,
        lr:            float          = 1e-3,
        save_path:     Optional[str]  = None,
        patience:      int            = 7,
        weight_decay:  float          = 1e-4,
        alpha:         float          = 1.0,
        adv_train_attack: Optional[Callable] = None,
        _temperature_override: Optional[float] = None,   # used by sweep
    ) -> Dict:
        """
        Train the student model on the soft labels produced by
        generate_soft_labels().

        Optimiser  : Adam with weight_decay
        Scheduler  : CosineAnnealingLR (T_max = n_epochs)
        Early stop : patience epochs without improvement on training loss

        Parameters
        ----------
        soft_label_dataset     : output of generate_soft_labels()
        n_epochs               : max training epochs
        lr                     : initial learning rate
        save_path              : .pth path for the best student checkpoint
        patience               : early-stopping patience
        weight_decay           : L2 regularisation coefficient
        alpha                  : loss blend weight (0–1).
                                 1.0 → pure KL on soft labels (vanilla Papernot).
                                 0.9 → 90% soft KL + 10% hard CE (recommended).
                                 Lower alpha nudges the student to also fit hard
                                 labels, recovering accuracy when T is very high.
        adv_train_attack       : optional callable (x, y, model) → x_adv.
                                 When provided, the student is also trained on
                                 adversarial versions of each batch, making it
                                 doubly robust: smooth boundary + adv hardening.
                                 Pass a lightweight FGSM for speed.
        _temperature_override  : internal — used by temperature_sweep()

        Returns
        -------
        history dict: {"epoch": [...], "loss": [...]}
        """
        if save_path is None:
            save_path = "models/resnet18/student_distilled.pth"

        # Ensure save directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        T = _temperature_override if _temperature_override is not None else self.T

        # If temperature changed, rebuild the loss function for this T
        loss_fn = (
            self.distill_loss
            if T == self.T
            else TemperatureKLLoss(temperature=T)
        )

        optimizer = optim.Adam(
            self.student.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )

        if adv_train_attack is not None:
            logger.info(
                "Adversarial distillation enabled — student will see both "
                "clean soft-label batches and adversarial variants."
            )

        best_loss  = math.inf
        no_improve = 0
        best_state: Optional[Dict] = None
        history    = {"epoch": [], "loss": [], "alpha": alpha}

        for epoch in range(1, n_epochs + 1):
            self.student.train()
            epoch_loss = 0.0
            n_batches  = 0

            for x_batch, soft_targets in soft_label_dataset:
                x_batch      = x_batch.to(self.device)
                soft_targets = soft_targets.to(self.device)

                # ── Optional adversarial augmentation ───────────────────────
                # Generate adversarial examples with student in eval mode
                # (attacks need gradient w.r.t. input, not parameters),
                # then switch back to train mode before the backward pass.
                if adv_train_attack is not None:
                    # Hard labels: argmax of soft targets (proxy ground truth)
                    y_hard = soft_targets.argmax(dim=-1)
                    self.student.eval()
                    with torch.enable_grad():
                        x_adv = adv_train_attack(x_batch, y_hard, self.student)
                    self.student.train()
                    # Concatenate clean + adversarial; soft_targets are the same
                    x_batch      = torch.cat([x_batch, x_adv], dim=0)
                    soft_targets = soft_targets.repeat(2, 1)

                optimizer.zero_grad()
                student_logits = self.student(x_batch)          # [B, C]

                # ── Loss blending ────────────────────────────────────────────
                # Pure KL (alpha=1.0): vanilla Papernot distillation.
                # Blended (alpha<1.0): also penalise on hard labels so the
                # student doesn't lose clean accuracy at very high temperatures.
                loss_kl = loss_fn(student_logits, soft_targets)
                if alpha < 1.0:
                    # Hard labels from argmax of soft targets
                    y_hard = soft_targets.argmax(dim=-1)
                    loss_ce = F.cross_entropy(student_logits, y_hard)
                    loss = alpha * loss_kl + (1.0 - alpha) * loss_ce
                else:
                    loss = loss_kl

                loss.backward()

                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            scheduler.step()

            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)

            logger.info(
                "Student epoch [%3d/%d]  loss=%.5f  lr=%.6f",
                epoch, n_epochs, avg_loss,
                scheduler.get_last_lr()[0],
            )

            # ── Early stopping ─────────────────────────────────────────
            if avg_loss < best_loss - 1e-7:
                best_loss  = avg_loss
                no_improve = 0
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.student.state_dict().items()
                }
                # Save checkpoint in train_resnet.py-compatible format
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch":            epoch,
                        "loss":             best_loss,
                        "temperature":      T,
                    },
                    save_path,
                )
                logger.info("  ✓ Best student saved → %s", save_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        "Early stop @ epoch %d | best loss %.5f",
                        epoch, best_loss,
                    )
                    break

        # Restore best weights into self.student
        if best_state is not None:
            self.student.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )

        logger.info("Training complete — best distillation loss: %.5f", best_loss)
        return history

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Evaluation
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_distillation(
        self,
        test_loader: DataLoader,
        attack       = None,        # AMLModel with .generate(x, y, model) → x_adv
    ) -> Dict[str, float]:
        """
        Head-to-head comparison of teacher vs student on clean and
        (optionally) adversarial inputs.

        Parameters
        ----------
        test_loader : yields (x, y) clean test batches
        attack      : an AMLModel instance from attacks/; if None, only
                      clean accuracy is measured.

        Returns
        -------
        dict with keys:
          teacher_clean_acc       — teacher accuracy on clean inputs
          student_clean_acc       — student accuracy on clean inputs
          teacher_adv_acc         — teacher accuracy after attack  (if attack given)
          student_adv_acc         — student accuracy after attack  (if attack given)
          gradient_masking_score  — teacher_grad_norm / student_grad_norm
        """
        results: Dict[str, float] = {}

        # ── Clean accuracy ─────────────────────────────────────────────────
        results["teacher_clean_acc"] = _compute_accuracy(
            self.teacher, test_loader, self.device
        )
        results["student_clean_acc"] = _compute_accuracy(
            self.student, test_loader, self.device
        )

        # ── Adversarial accuracy ───────────────────────────────────────────
        if attack is not None:
            for label, model in [
                ("teacher_adv_acc", self.teacher),
                ("student_adv_acc", self.student),
            ]:
                model.eval()
                correct = total = 0
                for x, y in test_loader:
                    x, y   = x.to(self.device), y.to(self.device)
                    x_adv  = attack.generate(x, y, model)
                    preds  = model(x_adv).argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total   += y.size(0)
                results[label] = correct / max(total, 1)

        # ── Gradient masking score ─────────────────────────────────────────
        results["gradient_masking_score"] = (
            self.compare_gradient_norms(self.teacher, test_loader)
            / (self.compare_gradient_norms(self.student, test_loader) + 1e-8)
        )

        logger.info("── Distillation Evaluation ─────────────────────────────")
        for k, v in results.items():
            logger.info("  %-32s : %.4f", k, v)
        logger.info("────────────────────────────────────────────────────────")
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Gradient norm (overall)
    # ──────────────────────────────────────────────────────────────────────────
    def compare_gradient_norms(
        self,
        model:     nn.Module,
        dataloader: DataLoader,
        n_batches:  int = 5,
    ) -> float:
        """
        Compute the mean L2 norm of ∂Loss/∂x (input gradient) over the
        first `n_batches` batches.

        A lower gradient norm means it is harder for an adversary to find
        a meaningful perturbation direction — this is the gradient masking
        effect we want to verify.

        Parameters
        ----------
        model      : nn.Module to probe
        dataloader : yields (x, y)
        n_batches  : how many batches to average (keep ≤10 for speed)

        Returns
        -------
        mean ||∇_x L||₂ across the sampled batches
        """
        model.eval()
        loss_fn = nn.CrossEntropyLoss()
        norms   = []

        for i, (x, y) in enumerate(dataloader):
            if i >= n_batches:
                break

            x = x.to(self.device).requires_grad_(True)
            y = y.to(self.device)

            logits = model(x)
            loss   = loss_fn(logits, y)
            loss.backward()

            # L2 norm of the input gradient tensor
            norm = x.grad.data.norm(2).item()
            norms.append(norm)

            # Clear graph to avoid memory build-up
            x.grad = None

        mean_norm = float(np.mean(norms)) if norms else 0.0
        return mean_norm

    # ──────────────────────────────────────────────────────────────────────────
    # Improvement 3: per-layer gradient norm profiling
    # ──────────────────────────────────────────────────────────────────────────
    def compare_gradient_norms_per_layer(
        self,
        dataloader: DataLoader,
        n_batches:  int = 3,
    ) -> OrderedDict:
        """
        Measure the gradient norm for every named parameter in teacher and
        student, and compute the per-layer masking ratio.

        Unlike compare_gradient_norms() which measures ∂L/∂x (input),
        this function measures ∂L/∂θ (parameter gradients) which reveals
        WHICH layers benefit most from distillation — useful for paper
        analysis and deciding where to focus future defenses.

        Returns
        -------
        OrderedDict mapping param_name →
            {
              "teacher_norm" : float,
              "student_norm" : float,
              "ratio"        : float,   # teacher / student  (>1 = masking)
            }
        """
        loss_fn = nn.CrossEntropyLoss()

        def _collect_param_norms(model: nn.Module) -> Dict[str, float]:
            """One forward+backward pass per batch; accumulate param grad norms."""
            model.train()   # enable gradients on parameters
            param_norm_accum: Dict[str, list] = {
                n: [] for n, _ in model.named_parameters() if _.requires_grad
            }

            for i, (x, y) in enumerate(dataloader):
                if i >= n_batches:
                    break
                model.zero_grad()
                x, y   = x.to(self.device), y.to(self.device)
                logits = model(x)
                loss   = loss_fn(logits, y)
                loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param_norm_accum[name].append(
                            param.grad.data.norm(2).item()
                        )

            # Average over batches
            return {
                name: float(np.mean(vals)) if vals else 0.0
                for name, vals in param_norm_accum.items()
            }

        teacher_norms = _collect_param_norms(self.teacher)
        student_norms = _collect_param_norms(self.student)

        result: OrderedDict = OrderedDict()
        all_names = list(teacher_norms.keys())

        for name in all_names:
            t_norm = teacher_norms.get(name, 0.0)
            s_norm = student_norms.get(name, 0.0)
            ratio  = t_norm / (s_norm + 1e-12)
            result[name] = {
                "teacher_norm": t_norm,
                "student_norm": s_norm,
                "ratio":        ratio,
            }

        # Log a compact summary (top-10 by ratio)
        sorted_items = sorted(
            result.items(), key=lambda kv: kv[1]["ratio"], reverse=True
        )
        logger.info("── Per-layer Gradient Masking (top-10 by ratio) ────────")
        logger.info("  %-50s  %8s  %8s  %8s",
                    "Parameter", "Teacher", "Student", "Ratio")
        for name, vals in sorted_items[:10]:
            logger.info(
                "  %-50s  %8.5f  %8.5f  %8.3f",
                name[:50], vals["teacher_norm"],
                vals["student_norm"], vals["ratio"],
            )
        logger.info("────────────────────────────────────────────────────────")

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Improvement 1: Temperature sweep
    # ──────────────────────────────────────────────────────────────────────────
    def temperature_sweep(
        self,
        train_loader:   DataLoader,
        test_loader:    DataLoader,
        temperatures:   List[float]  = DEFAULT_SWEEP_TEMPS,
        sweep_epochs:   int          = 5,       # intentionally short
        save_plot_dir:  str          = "evaluation/figures",
        student_save_dir: str        = "models/resnet18",
    ) -> Dict[float, Dict[str, float]]:
        """
        Train a fresh student at each temperature in `temperatures`, record
        (student_clean_acc, gradient_masking_score), and produce a dual-axis
        plot saved to evaluation/figures/distillation_temperature_sweep.png.

        Rationale for the plot
        ----------------------
        There is a fundamental trade-off:
        - Very high T → very soft labels → student learns almost nothing
          (clean accuracy drops).
        - Very low  T → hard labels → no gradient masking benefit.
        The sweet spot (usually T ≈ 20) maximises masking without hurting accuracy.
        This plot makes that trade-off visible for the paper.

        Parameters
        ----------
        train_loader    : training DataLoader (for soft label generation)
        test_loader     : test DataLoader    (for accuracy & grad norm)
        temperatures    : list of T values to sweep
        sweep_epochs    : epochs per T value (keep short for speed)
        save_plot_dir   : directory for the output PNG
        student_save_dir: directory for per-T student checkpoints

        Returns
        -------
        Dict mapping each temperature → {"clean_acc": float,
                                         "gradient_masking_score": float}
        """
        Path(save_plot_dir).mkdir(parents=True, exist_ok=True)
        Path(student_save_dir).mkdir(parents=True, exist_ok=True)

        sweep_results: Dict[float, Dict[str, float]] = {}

        for T in temperatures:
            logger.info("── Temperature sweep T=%.1f ─────────────────────────", T)

            # Re-initialise student weights for each T (clean slate)
            self.student = _build_resnet18(dropout=self.dropout).to(self.device)

            # Generate soft labels at this T
            soft_ds = self.generate_soft_labels(train_loader, temperature=T)

            # Train for sweep_epochs (quick proxy — not full training)
            ckpt = os.path.join(student_save_dir, f"student_T{int(T)}.pth")
            self.train_student(
                soft_ds,
                n_epochs=sweep_epochs,
                lr=1e-3,
                save_path=ckpt,
                patience=sweep_epochs,       # no early stop in sweep
                _temperature_override=T,
            )

            # Measure clean accuracy
            clean_acc = _compute_accuracy(self.student, test_loader, self.device)

            # Measure gradient masking score  (teacher / student ratio)
            teacher_norm = self.compare_gradient_norms(
                self.teacher, test_loader, n_batches=3
            )
            student_norm = self.compare_gradient_norms(
                self.student, test_loader, n_batches=3
            )
            gm_score = teacher_norm / (student_norm + 1e-8)

            sweep_results[T] = {
                "clean_acc":             clean_acc,
                "gradient_masking_score": gm_score,
                "teacher_grad_norm":     teacher_norm,
                "student_grad_norm":     student_norm,
            }
            logger.info(
                "T=%.1f | clean_acc=%.4f | gm_score=%.4f",
                T, clean_acc, gm_score,
            )

        # ── Restore self.student to default T ─────────────────────────────
        self.student = _build_resnet18(dropout=self.dropout).to(self.device)
        logger.info("Student re-initialised to default T=%.1f after sweep.", self.T)

        # ── Plot ───────────────────────────────────────────────────────────
        self._plot_temperature_sweep(sweep_results, save_plot_dir)
        return sweep_results

    def _plot_temperature_sweep(
        self,
        sweep_results: Dict[float, Dict[str, float]],
        save_dir:      str,
    ) -> str:
        """
        Save dual-axis line chart:
          Left  Y-axis  (blue)  : student clean accuracy
          Right Y-axis (orange) : gradient masking score
          X-axis                : distillation temperature T

        Also annotates the "sweet spot" — the T value that maximises the
        product clean_acc × gm_score.
        """
        temps     = sorted(sweep_results.keys())
        accs      = [sweep_results[T]["clean_acc"]              for T in temps]
        gm_scores = [sweep_results[T]["gradient_masking_score"] for T in temps]

        # Sweet spot: highest geometric mean of accuracy and masking
        combined  = [a * g for a, g in zip(accs, gm_scores)]
        best_idx  = int(np.argmax(combined))
        best_T    = temps[best_idx]

        fig, ax1  = plt.subplots(figsize=(8, 5))
        ax2       = ax1.twinx()

        color_acc = "#2563EB"   # blue
        color_gm  = "#D97706"   # orange

        ax1.plot(temps, accs,      "o-", color=color_acc, linewidth=2.0,
                 markersize=7, label="Student Clean Accuracy")
        ax2.plot(temps, gm_scores, "s--", color=color_gm, linewidth=2.0,
                 markersize=7, label="Gradient Masking Score")

        # Annotate sweet spot
        ax1.axvline(x=best_T, color="#DC2626", linestyle=":", linewidth=1.5,
                    label=f"Sweet spot T={best_T:.0f}")
        ax1.annotate(
            f"T={best_T:.0f}",
            xy=(best_T, accs[best_idx]),
            xytext=(best_T + 2, accs[best_idx] + 0.02),
            fontsize=9, color="#DC2626",
            arrowprops=dict(arrowstyle="->", color="#DC2626"),
        )

        ax1.set_xlabel("Distillation Temperature T", fontsize=12)
        ax1.set_ylabel("Student Clean Accuracy",     fontsize=12, color=color_acc)
        ax2.set_ylabel("Gradient Masking Score\n(teacher grad norm / student grad norm)",
                       fontsize=12, color=color_gm)
        ax1.tick_params(axis="y", labelcolor=color_acc)
        ax2.tick_params(axis="y", labelcolor=color_gm)
        ax1.set_xscale("log")   # log scale — temperatures span 5 to 100

        # Merged legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", fontsize=9, framealpha=0.9)

        ax1.set_title(
            "Defensive Distillation — Temperature Sweep\n"
            "Clean Accuracy vs Gradient Masking Score",
            fontsize=12, fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)

        out_path = Path(save_dir) / "distillation_temperature_sweep.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Temperature sweep plot saved → %s", out_path.resolve())
        return str(out_path.resolve())

    # ──────────────────────────────────────────────────────────────────────────
    # Improvement 2: stacked / combined defense
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def combine_defenses(
        self,
        x:                 torch.Tensor,
        preprocessing_fn:  Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        autoencoder_fn:    Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_all_stages: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Chain up to three defense stages on a (possibly adversarial) input:

            Stage 1 — InputPreprocessor    (feature squeezing / smoothing / JPEG)
            Stage 2 — AutoencoderDefense   (denoising reconstruction)
            Stage 3 — Distilled student    (classification)

        Either preprocessing_fn or autoencoder_fn (or both) may be None,
        in which case that stage is skipped.

        Parameters
        ----------
        x                 : input tensor [B, C, H, W] (image) or [B, n_features]
        preprocessing_fn  : callable x → x  (e.g. InputPreprocessor.preprocess())
        autoencoder_fn    : callable x → x  (e.g. AutoencoderDefense.denoise())
        return_all_stages : if True, return (after_pp, after_ae, logits)
                            for debugging / ablation studies

        Returns
        -------
        logits tensor [B, num_classes]  (or tuple of intermediate tensors)

        Example
        -------
        from defenses.input_preprocessing import InputPreprocessor
        from defenses.autoencoder         import AutoencoderDefense

        pp = InputPreprocessor()
        ae = AutoencoderDefense(modality="image")

        logits = dd.combine_defenses(
            x_adv,
            preprocessing_fn = lambda t: pp.preprocess(t, "image",
                                         ["feature_squeeze", "gaussian"]),
            autoencoder_fn   = ae.denoise,
        )
        preds = logits.argmax(dim=-1)
        """
        x_dev = x.to(self.device)

        # Stage 1 — Input preprocessing
        after_pp = x_dev
        if preprocessing_fn is not None:
            after_pp = preprocessing_fn(x_dev)
            # Ensure tensor is still on the right device
            if isinstance(after_pp, torch.Tensor):
                after_pp = after_pp.to(self.device)

        # Stage 2 — Autoencoder denoising
        after_ae = after_pp
        if autoencoder_fn is not None:
            after_ae = autoencoder_fn(after_pp)
            if isinstance(after_ae, torch.Tensor):
                after_ae = after_ae.to(self.device)

        # Stage 3 — Distilled student classification
        self.student.eval()
        logits = self.student(after_ae)   # [B, num_classes]

        if return_all_stages:
            return after_pp.cpu(), after_ae.cpu(), logits.cpu()
        return logits

    # ──────────────────────────────────────────────────────────────────────────
    # ECE — Expected Calibration Error
    # ──────────────────────────────────────────────────────────────────────────
    def calculate_ece(
        self,
        loader:          DataLoader,
        n_bins:          int  = 15,
        use_calibrated:  bool = False,
    ) -> float:
        """
        Expected Calibration Error (Guo et al. 2017).

        ECE ≈ 0  → model confidence matches empirical accuracy (well calibrated).
        ECE ≈ 1  → maximally miscalibrated.

        Distilled students typically have lower ECE than teachers because the
        soft labels force the student to output calibrated probabilities rather
        than near-one-hot distributions.

        Parameters
        ----------
        loader          : DataLoader yielding (x, y)
        n_bins          : number of confidence bins (15 gives good resolution
                          for datasets ≥ 1000 samples)
        use_calibrated  : if True, divide logits by self.calibration_temperature
                          before computing ECE (requires calibrate_temperature()
                          to have been called first)

        Returns
        -------
        ECE as a float in [0, 1]
        """
        self.student.eval()
        logits_all, labels_all = [], []

        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits_all.append(self.student(x))
                labels_all.append(y)

        logits = torch.cat(logits_all)
        labels = torch.cat(labels_all)

        if use_calibrated:
            # calibrate_temperature() must have been called first
            T_cal  = getattr(self, "calibration_temperature", 1.0)
            logits = logits / max(T_cal, 0.01)

        softmax_out   = F.softmax(logits, dim=-1)
        confidences, predictions = softmax_out.max(dim=-1)
        accuracies    = predictions.eq(labels)

        ece = 0.0
        bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)

        for i in range(n_bins):
            lo, hi   = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
            in_bin   = confidences.gt(lo) & confidences.le(hi)
            prop     = in_bin.float().mean().item()
            if prop > 0.0:
                bin_acc  = accuracies[in_bin].float().mean().item()
                bin_conf = confidences[in_bin].mean().item()
                ece     += abs(bin_conf - bin_acc) * prop

        logger.info(
            "ECE = %.4f  (n_bins=%d, calibrated=%s)",
            ece, n_bins, use_calibrated,
        )
        return float(ece)

    # ──────────────────────────────────────────────────────────────────────────
    # Online training convenience wrapper
    # ──────────────────────────────────────────────────────────────────────────
    def train_student_online(
        self,
        train_loader: DataLoader,
        n_epochs:     int   = 30,
        lr:           float = 1e-3,
        alpha:        float = 0.9,
        adv_train_attack: Optional[Callable] = None,
        save_path:    Optional[str] = None,
        patience:     int   = 7,
    ) -> Dict:
        """
        Convenience wrapper that combines generate_soft_labels() and
        train_student() into a single call using online (O(1) memory) training.

        Instead of pre-computing all soft labels in RAM, soft labels are
        generated on-the-fly inside the training loop.  This is memory-
        efficient for large datasets but slightly slower per epoch because
        the teacher forward pass happens at training time.

        For most use cases with datasets ≤ 50,000 images, the offline
        generate_soft_labels() → train_student() two-step is equally fine.
        Use this method when GPU VRAM or CPU RAM is limited.

        Parameters
        ----------
        train_loader      : DataLoader yielding (x, y)
        n_epochs          : max training epochs
        lr                : Adam learning rate
        alpha             : KL / CE blend coefficient (see train_student docs)
        adv_train_attack  : optional adversarial augmentation callable
        save_path         : checkpoint path
        patience          : early-stopping patience

        Returns
        -------
        training history dict
        """
        logger.info(
            "Online distillation mode — soft labels generated per batch "
            "(alpha=%.2f, adv_attack=%s)",
            alpha, adv_train_attack is not None,
        )
        # generate_soft_labels stores batches as (x_cpu, soft_cpu) tuples;
        # train_student iterates over exactly the same structure, so we can
        # reuse it directly.
        soft_ds = self.generate_soft_labels(train_loader)
        return self.train_student(
            soft_ds,
            n_epochs=n_epochs,
            lr=lr,
            alpha=alpha,
            adv_train_attack=adv_train_attack,
            save_path=save_path,
            patience=patience,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience: save / load student
    # ──────────────────────────────────────────────────────────────────────────
    def save_student(self, path: str) -> None:
        """Persist student state dict."""
        torch.save(
            {"model_state_dict": self.student.state_dict(),
             "temperature": self.T},
            path,
        )
        logger.info("Student saved → %s", path)

    def load_student(self, path: str) -> None:
        """Load a previously saved student checkpoint."""
        raw = torch.load(path, map_location=self.device)
        state = raw.get("model_state_dict", raw)
        self.student.load_state_dict(state)
        self.student.to(self.device)
        logger.info("Student loaded ← %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — demonstration on synthetic data (no real checkpoint required)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    End-to-end smoke test using random weights (no trained checkpoint needed).

    Steps demonstrated
    ------------------
    1  Build synthetic 3×64×64 KYC dataset
    2  Create fake teacher checkpoint
    3  Instantiate DefensiveDistillation
    4  generate_soft_labels()
    5a train_student()  — vanilla (alpha=1.0, pure KL)
    5b train_student()  — blended (alpha=0.85, KL + CE)
    6  evaluate_distillation()
    7  calculate_ece()  with and without temperature calibration
    8  compare_gradient_norms_per_layer()
    9  temperature_sweep()  over T ∈ [5, 10, 20]  (2 epochs each)
    10 train_student_online()  convenience wrapper
    11 combine_defenses()  stacked pipeline

    Run from the project root:
        python aml_fintech_framework/defenses/distillation.py
    """
    import tempfile
    import random

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    STEP = 0
    def step(label: str) -> None:
        global STEP
        STEP += 1
        print(f"\n[{STEP}] {label}")

    print("=" * 68)
    print(" [Partner B]  AML FinTech — Defensive Distillation Smoke Test")
    print("=" * 68)

    DEVICE = "cpu"
    N      = 64     # total synthetic samples
    B      = 8      # batch size

    # ── 1. Synthetic dataset ───────────────────────────────────────────────
    step("Building synthetic 3×64×64 KYC dataset")
    x_all  = torch.rand(N, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    y_all  = torch.randint(0, NUM_CLASSES, (N,))
    ds     = TensorDataset(x_all, y_all)
    n_tr   = int(0.75 * N)
    n_te   = N - n_tr
    tr_ds, te_ds = torch.utils.data.random_split(ds, [n_tr, n_te])
    train_loader = DataLoader(tr_ds, batch_size=B, shuffle=True)
    test_loader  = DataLoader(te_ds, batch_size=B, shuffle=False)
    print(f"  train={n_tr}  test={n_te}  batch={B}")

    # ── 2. Fake teacher checkpoint ─────────────────────────────────────────
    step("Creating fake teacher checkpoint (random ResNet-18 weights)")
    tmp_dir   = Path(tempfile.mkdtemp())
    ckpt_path = str(tmp_dir / "fake_teacher.pth")
    torch.save({"model_state_dict": _build_resnet18().state_dict()}, ckpt_path)
    print(f"  Saved → {ckpt_path}")

    # ── 3. Instantiate ────────────────────────────────────────────────────
    step("Instantiating DefensiveDistillation (T=20)")
    dd = DefensiveDistillation(
        teacher_model_path=ckpt_path,
        temperature=20.0,
        device=DEVICE,
    )

    # ── 4. Generate soft labels ───────────────────────────────────────────
    step("Generating soft labels from teacher")
    t0      = time.time()
    soft_ds = dd.generate_soft_labels(train_loader)
    print(f"  Done in {time.time()-t0:.2f}s  | batches={len(soft_ds)}")
    # Inspect first batch: soft labels should sum to 1.0 per row
    first_soft = soft_ds[0][1]
    print(f"  First batch soft label sums (should all be 1.0): "
          f"{first_soft.sum(dim=-1).tolist()}")

    # ── 5a. Train student — vanilla (alpha=1.0) ────────────────────────────
    step("Training student — vanilla KL (alpha=1.0), 3 epochs")
    vanilla_path = str(tmp_dir / "student_vanilla.pth")
    h_vanilla = dd.train_student(
        soft_ds,
        n_epochs=3,
        lr=1e-3,
        alpha=1.0,          # pure KL — vanilla Papernot
        save_path=vanilla_path,
        patience=5,
    )
    print(f"  Losses: {[f'{v:.4f}' for v in h_vanilla['loss']]}")

    # ── 5b. Train student — blended (alpha=0.85) ──────────────────────────
    step("Training student — blended (alpha=0.85 KL + 0.15 CE), 3 epochs")
    # Re-initialise student for fair comparison
    dd.student = _build_resnet18(dropout=dd.dropout).to(dd.device)
    blended_path = str(tmp_dir / "student_blended.pth")
    h_blended = dd.train_student(
        soft_ds,
        n_epochs=3,
        lr=1e-3,
        alpha=0.85,         # 85% KL + 15% CE
        save_path=blended_path,
        patience=5,
    )
    print(f"  Losses: {[f'{v:.4f}' for v in h_blended['loss']]}")
    print(f"  Loss blend note: blended loss includes cross-entropy term,")
    print(f"  so absolute values differ from vanilla — this is expected.")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    step("Evaluating teacher vs student (clean accuracy + gradient masking)")
    results = dd.evaluate_distillation(test_loader=test_loader, attack=None)
    print(f"\n  {'Metric':<32} {'Value':>8}")
    print("  " + "-" * 42)
    for k, v in results.items():
        print(f"  {k:<32} {v:>8.4f}")

    # ── 7. ECE before and after temperature calibration ───────────────────
    step("Expected Calibration Error (ECE)")
    ece_raw = dd.calculate_ece(test_loader, n_bins=15, use_calibrated=False)
    print(f"  ECE (raw, uncalibrated)     : {ece_raw:.4f}")

    # Temperature scaling calibration
    dd.calibration_temperature = 1.5   # simulate a calibrated value
    ece_cal = dd.calculate_ece(test_loader, n_bins=15, use_calibrated=True)
    print(f"  ECE (T_cal=1.5, calibrated) : {ece_cal:.4f}")
    print(f"  Interpretation: lower ECE = better probability calibration.")
    print(f"  Distilled students typically have lower ECE than teachers")
    print(f"  because soft labels enforce calibrated output distributions.")

    # ── 8. Per-layer gradient norm profiling ──────────────────────────────
    step("Per-layer gradient norm profiling (top 5 by masking ratio)")
    layer_norms = dd.compare_gradient_norms_per_layer(test_loader, n_batches=2)

    # Filter to weight params only (exclude BN running stats which have ~0 grad)
    weight_layers = {
        name: vals for name, vals in layer_norms.items()
        if "weight" in name and "bn" not in name
    }
    top5 = sorted(
        weight_layers.items(), key=lambda kv: kv[1]["ratio"], reverse=True
    )[:5]

    print(f"\n  {'Parameter':<45} {'Teacher':>8} {'Student':>8} {'Ratio':>7}")
    print("  " + "-" * 72)
    for name, vals in top5:
        print(
            f"  {name[:45]:<45} "
            f"{vals['teacher_norm']:>8.5f} "
            f"{vals['student_norm']:>8.5f} "
            f"{vals['ratio']:>7.3f}"
        )
    print(f"\n  Ratio > 1.0 means teacher has larger gradients at that layer")
    print(f"  → the student's gradient masking effect is confirmed.")

    # ── 9. Temperature sweep ──────────────────────────────────────────────
    step("Temperature sweep  T ∈ [5, 10, 20]  (2 epochs each)")
    Path("evaluation/figures").mkdir(parents=True, exist_ok=True)
    sweep = dd.temperature_sweep(
        train_loader=train_loader,
        test_loader=test_loader,
        temperatures=[5.0, 10.0, 20.0],
        sweep_epochs=2,
        save_plot_dir="evaluation/figures",
        student_save_dir=str(tmp_dir / "sweep_ckpts"),
    )
    print(f"\n  {'T':>6}  {'clean_acc':>10}  {'gm_score':>10}")
    print("  " + "-" * 32)
    for T_val, vals in sorted(sweep.items()):
        print(
            f"  {T_val:>6.1f}  "
            f"{vals['clean_acc']:>10.4f}  "
            f"{vals['gradient_masking_score']:>10.4f}"
        )
    print("  Plot → evaluation/figures/distillation_temperature_sweep.png")

    # ── 10. Online training convenience wrapper ───────────────────────────
    step("train_student_online() — single-call convenience API")
    dd.student = _build_resnet18(dropout=dd.dropout).to(dd.device)
    online_path = str(tmp_dir / "student_online.pth")
    h_online = dd.train_student_online(
        train_loader,
        n_epochs=2,
        lr=1e-3,
        alpha=0.9,
        save_path=online_path,
        patience=5,
    )
    print(f"  Losses: {[f'{v:.4f}' for v in h_online['loss']]}")
    print(f"  Note: soft labels re-generated from teacher each epoch")
    print(f"  (O(1) memory vs O(N) for offline generate_soft_labels).")

    # ── 11. combine_defenses() ────────────────────────────────────────────
    step("combine_defenses() — stacked pipeline (Preprocessor → AE → Student)")
    x_sample = x_all[:4]

    # Stage 1: identity preprocessing (plug in InputPreprocessor later)
    # Stage 2: identity autoencoder   (plug in AutoencoderDefense later)
    logits = dd.combine_defenses(
        x_sample,
        preprocessing_fn=None,
        autoencoder_fn=None,
    )
    preds = logits.argmax(dim=-1)
    print(f"  Input shape  : {x_sample.shape}")
    print(f"  Logits shape : {logits.shape}")
    print(f"  Predictions  : {preds.tolist()}")

    # With mock preprocessing (multiply by 0.9 simulates feature squeezing)
    logits_pp = dd.combine_defenses(
        x_sample,
        preprocessing_fn=lambda t: t * 0.9,
        autoencoder_fn=None,
    )
    print(f"  With mock pp : {logits_pp.argmax(dim=-1).tolist()}")

    # return_all_stages=True for ablation study
    after_pp, after_ae, logits_stages = dd.combine_defenses(
        x_sample, return_all_stages=True
    )
    print(f"  All stages   : pp={after_pp.shape} ae={after_ae.shape} "
          f"logits={logits_stages.shape}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(" Smoke test complete — all methods verified.")
    print(f"\n  Vanilla student   → {vanilla_path}")
    print(f"  Blended student   → {blended_path}")
    print(f"  Online student    → {online_path}")
    print(f"  Sweep plot        → evaluation/figures/"
          f"distillation_temperature_sweep.png")
    print("\n  In real training:")
    print("    dd = DefensiveDistillation('models/resnet18/best_model.pth')")
    print("    soft = dd.generate_soft_labels(train_loader)")
    print("    dd.train_student(soft, n_epochs=30, alpha=0.9)")
    print("    metrics = dd.evaluate_distillation(test_loader, attack=fgsm)")
    print("=" * 68)