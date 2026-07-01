# =============================================================================
# attacks/base.py
# [Partner A]  — Abstract base class + shared data-classes for all attacks.
#
# Every concrete attack (FGSM, PGD, C&W, DeepFool) must subclass AMLModel and
# implement generate().  AttackConfig bundles hyper-parameters; AttackResult
# carries everything needed for logging, evaluation, and the LaTeX tables.
# =============================================================================

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass — passed into every attack at construction time
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AttackConfig:
    """Hyper-parameter bundle shared by all attack implementations."""
    epsilon: float = 0.3            # L-inf perturbation budget
    norm: str = "linf"              # "linf" | "l2"
    targeted: bool = False          # targeted (minimise loss to target) vs untargeted
    target_class: Optional[int] = None
    clip_min: float = 0.0           # valid input range lower bound
    clip_max: float = 1.0           # valid input range upper bound
    verbose: bool = False           # print per-step debug info


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass — every generate() call returns one of these
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AttackResult:
    """Holds the original sample, the adversarial sample, and all metrics."""
    x_orig: torch.Tensor            # original input
    x_adv: torch.Tensor             # adversarial (perturbed) input
    y_true: torch.Tensor            # ground-truth labels
    y_pred_orig: torch.Tensor       # model predictions on x_orig
    y_pred_adv: torch.Tensor        # model predictions on x_adv
    perturbation_norm: float        # L2 norm of (x_adv - x_orig)
    success: bool                   # True when at least one sample is misclassified
    attack_name: str = "unknown"
    epsilon: float = 0.0
    iterations: int = 0
    modality: str = "unknown"
    extra: dict = field(default_factory=dict)  # adapter-specific metadata

    # ------------------------------------------------------------------
    # Convenience properties used by evaluators and LaTeX table generator
    # ------------------------------------------------------------------
    @property
    def attack_success_rate(self) -> float:
        """Fraction of originally-correct samples that were fooled."""
        correct_mask = (self.y_pred_orig == self.y_true)
        if correct_mask.sum().item() == 0:
            return 0.0
        fooled = ((self.y_pred_adv != self.y_true) & correct_mask).sum().item()
        return fooled / correct_mask.sum().item()

    @property
    def adversarial_accuracy(self) -> float:
        """Accuracy of the model on adversarial examples."""
        return (self.y_pred_adv == self.y_true).float().mean().item()

    @property
    def clean_accuracy(self) -> float:
        """Accuracy of the model on clean examples."""
        return (self.y_pred_orig == self.y_true).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────
class AMLModel(ABC):
    """
    Abstract base class for all adversarial attack implementations.

    Subclasses must implement generate() and __repr__().
    Concrete helper methods (clip, l2_norm, …) are provided here so every
    attack automatically inherits them without code duplication.
    """

    def __init__(self, config: AttackConfig):
        self.config = config

    # ── Must override ────────────────────────────────────────────────────────
    @abstractmethod
    def generate(self, x: torch.Tensor, y: torch.Tensor, model: nn.Module,
                 **kwargs) -> AttackResult:
        """Generate adversarial examples from clean inputs x with labels y."""
        ...

    @abstractmethod
    def __repr__(self) -> str: ...

    # ── Shared geometry helpers ──────────────────────────────────────────────
    def clip(self, x: torch.Tensor, x_orig: torch.Tensor,
             epsilon: float) -> torch.Tensor:
        """Clip x so ||x - x_orig||_inf ≤ epsilon AND x ∈ [clip_min, clip_max]."""
        x_clipped = torch.clamp(x, x_orig - epsilon, x_orig + epsilon)
        return torch.clamp(x_clipped, self.config.clip_min, self.config.clip_max)

    def l2_norm(self, delta: torch.Tensor) -> float:
        """Return the scalar L2 norm of a perturbation tensor."""
        return delta.norm(p=2).item()

    def linf_norm(self, delta: torch.Tensor) -> float:
        """Return the scalar L-inf norm of a perturbation tensor."""
        return delta.abs().max().item()

    def project(self, delta: torch.Tensor, norm: str,
                epsilon: float) -> torch.Tensor:
        """Project delta onto the norm ball of radius epsilon."""
        if norm == "linf":
            return delta.clamp(-epsilon, epsilon)
        elif norm == "l2":
            # Scale down if outside the L2 ball
            nrm = delta.norm(p=2)
            if nrm > epsilon:
                delta = delta * (epsilon / (nrm + 1e-12))
            return delta
        else:
            raise ValueError(f"Unsupported norm '{norm}'. Use 'linf' or 'l2'.")


# ─────────────────────────────────────────────────────────────────────────────
# Utility function used by evaluators
# ─────────────────────────────────────────────────────────────────────────────
def compute_attack_success_rate(results: List[AttackResult]) -> float:
    """
    Aggregate attack success rate across a list of AttackResult objects.
    Returns the fraction of originally-correct samples that were fooled.
    """
    total_correct = 0
    total_fooled = 0
    for r in results:
        correct_mask = (r.y_pred_orig == r.y_true)
        total_correct += correct_mask.sum().item()
        total_fooled += ((r.y_pred_adv != r.y_true) & correct_mask).sum().item()
    if total_correct == 0:
        return 0.0
    return total_fooled / total_correct