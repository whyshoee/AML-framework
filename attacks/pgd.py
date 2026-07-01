# =============================================================================
# attacks/pgd.py
# [Partner A]  — Projected Gradient Descent (Madry et al., 2018)
#
# Iterative FGSM with projection back onto the L-inf ε-ball after each step:
#   x^{t+1} = Π_{x+S} ( x^t + α · sign(∇_x J) )
# Considered the gold standard for evaluating adversarial robustness.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from attacks.base import AMLModel, AttackConfig, AttackResult


class PGD(AMLModel):
    """Iterative projected gradient descent attack (L-inf ball)."""

    def __init__(self, config: AttackConfig | None = None,
                 n_steps: int = 40, step_size: float = 0.01,
                 random_start: bool = True):
        super().__init__(config or AttackConfig())
        self.n_steps     = n_steps
        self.step_size   = step_size
        self.random_start = random_start

    def __repr__(self) -> str:
        return (f"PGD(epsilon={self.config.epsilon}, "
                f"n_steps={self.n_steps}, step_size={self.step_size})")

    def generate(self, x: torch.Tensor, y: torch.Tensor,
                 model: nn.Module, loss_fn=None, **kwargs) -> AttackResult:
        loss_fn = loss_fn or nn.CrossEntropyLoss()
        model.eval()
        eps = self.config.epsilon

        # ── optional random initialisation inside ε-ball ─────────────────────
        if self.random_start:
            noise = torch.empty_like(x).uniform_(-eps, eps)
            x_adv = torch.clamp(x + noise,
                                 self.config.clip_min, self.config.clip_max).detach()
        else:
            x_adv = x.clone().detach()

        # ── PGD loop ─────────────────────────────────────────────────────────
        for _ in range(self.n_steps):
            x_adv = x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss   = loss_fn(logits, y)
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + self.step_size * x_adv.grad.sign()
                # project back onto ε-ball centred on x
                delta = torch.clamp(x_adv - x, -eps, eps)
                x_adv = torch.clamp(x + delta,
                                     self.config.clip_min, self.config.clip_max)

        # ── evaluate ─────────────────────────────────────────────────────────
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
            attack_name="PGD",
            epsilon=eps,
            iterations=self.n_steps,
        )