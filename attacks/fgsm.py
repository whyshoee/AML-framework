# =============================================================================
# attacks/fgsm.py
# [Partner A]  — Fast Gradient Sign Method (Goodfellow et al., 2015)
#
# x_adv = x + ε · sign(∇_x J(θ, x, y))
#
# One forward + one backward pass; cheapest gradient-based attack.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from attacks.base import AMLModel, AttackConfig, AttackResult


class FGSM(AMLModel):
    """Single-step gradient sign attack."""

    def __init__(self, config: AttackConfig | None = None):
        super().__init__(config or AttackConfig())

    def __repr__(self) -> str:
        return f"FGSM(epsilon={self.config.epsilon})"

    def generate(self, x: torch.Tensor, y: torch.Tensor,
                 model: nn.Module, loss_fn=None, **kwargs) -> AttackResult:
        loss_fn = loss_fn or nn.CrossEntropyLoss()
        model.eval()

        x_adv = x.clone().detach().requires_grad_(True)

        # ── forward pass ────────────────────────────────────────────────────
        logits = model(x_adv)
        if self.config.targeted and self.config.target_class is not None:
            t = torch.full_like(y, self.config.target_class)
            loss = -loss_fn(logits, t)           # minimise loss toward target
        else:
            loss = loss_fn(logits, y)             # maximise loss from true class

        # ── backward pass — gradient w.r.t. input ───────────────────────────
        model.zero_grad()
        loss.backward()

        # x_adv = x + ε · sign(∇_x loss)
        with torch.no_grad():
            x_adv = x + self.config.epsilon * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, self.config.clip_min, self.config.clip_max)

        # ── evaluate original and adversarial predictions ────────────────────
        with torch.no_grad():
            y_pred_orig = model(x).argmax(dim=1)
            y_pred_adv  = model(x_adv).argmax(dim=1)

        delta = x_adv - x
        return AttackResult(
            x_orig=x.detach(),
            x_adv=x_adv.detach(),
            y_true=y,
            y_pred_orig=y_pred_orig,
            y_pred_adv=y_pred_adv,
            perturbation_norm=self.l2_norm(delta),
            success=(y_pred_adv != y).any().item(),
            attack_name="FGSM",
            epsilon=self.config.epsilon,
            iterations=1,
        )