# =============================================================================
# attacks/cw.py
# [Partner A]  — Carlini & Wagner L2 Attack (Carlini & Wagner, 2017)
#
# Optimises in tanh-space to enforce box constraints without projection:
#   w = arctanh(2x - 1),  x_adv = 0.5(tanh(w+δ) + 1)
#
# Objective: ||δ||_2^2 + c · f(x_adv)
#   where f(x_adv) = max( Z[y_true] - max_{i≠y_true} Z[i] + κ, 0 )
#
# Binary search over c finds the smallest perturbation that causes misclassification.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from attacks.base import AMLModel, AttackConfig, AttackResult


class CarliniWagner(AMLModel):
    """C&W L2 attack with binary search over the regularisation constant c."""

    def __init__(self, config: AttackConfig | None = None,
                 c: float = 1e-4, kappa: float = 0.0,
                 n_steps: int = 1000, lr: float = 0.01,
                 binary_search_steps: int = 9):
        super().__init__(config or AttackConfig())
        self.c_init              = c
        self.kappa               = kappa          # confidence margin
        self.n_steps             = n_steps
        self.lr                  = lr
        self.binary_search_steps = binary_search_steps

    def __repr__(self) -> str:
        return (f"CarliniWagner(c={self.c_init}, kappa={self.kappa}, "
                f"n_steps={self.n_steps})")

    # ── internal objective f(x_adv) ──────────────────────────────────────────
    def _f(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        f(x_adv) = max( Z[y_true] - max_{i≠y_true} Z[i] + κ, 0 )
        A positive value means the sample is NOT yet misclassified.
        """
        n = logits.size(0)
        correct_logit = logits[range(n), y]                        # Z[y_true]
        # mask the true class so we take the max over wrong classes
        logits_masked = logits.clone()
        logits_masked[range(n), y] = -1e9
        max_wrong_logit = logits_masked.max(dim=1).values          # max_{i≠y}
        return torch.clamp(correct_logit - max_wrong_logit + self.kappa, min=0.0)

    def generate(self, x: torch.Tensor, y: torch.Tensor,
                 model: nn.Module, **kwargs) -> AttackResult:
        model.eval()
        batch_size = x.size(0)
        best_adv   = x.clone().detach()
        best_norm  = torch.full((batch_size,), float("inf"))

        # Binary search bounds for c (one value per sample)
        c_lower = torch.zeros(batch_size)
        c_upper = torch.full((batch_size,), 1e10)
        c       = torch.full((batch_size,), self.c_init)

        # Change-of-variables: w s.t. x = 0.5*(tanh(w)+1) ∈ [0,1]
        x_tanh = torch.atanh((x.clone() * 2 - 1).clamp(-1 + 1e-6, 1 - 1e-6))

        for _ in range(self.binary_search_steps):
            # Optimise δ in tanh-space
            delta_w = torch.zeros_like(x_tanh, requires_grad=True)
            optimizer = optim.Adam([delta_w], lr=self.lr)

            for step in range(self.n_steps):
                optimizer.zero_grad()
                x_adv = 0.5 * (torch.tanh(x_tanh + delta_w) + 1)  # box-constrained

                # L2 term: ||x_adv - x||_2^2
                l2_loss = (x_adv - x).flatten(1).norm(p=2, dim=1).pow(2)

                # Adversarial term: c · f(x_adv)
                logits = model(x_adv)
                adv_loss = self._f(logits, y)

                c_device = c.to(x.device)
                loss = (l2_loss + c_device * adv_loss).sum()
                loss.backward()
                optimizer.step()

            # ── update best adversarial examples after each binary search step ──
            with torch.no_grad():
                x_adv_final = 0.5 * (torch.tanh(x_tanh + delta_w) + 1)
                logits_final = model(x_adv_final)
                preds_final  = logits_final.argmax(dim=1)
                l2_norms     = (x_adv_final - x).flatten(1).norm(p=2, dim=1)

                for i in range(batch_size):
                    if preds_final[i] != y[i] and l2_norms[i] < best_norm[i]:
                        best_norm[i] = l2_norms[i].item()
                        best_adv[i]  = x_adv_final[i].detach()
                        # tighten binary search: found → lower c
                        c_upper[i] = c[i].item()
                    else:
                        # not found → raise c
                        c_lower[i] = c[i].item()

                # update c toward the midpoint
                c = (c_lower + c_upper) / 2.0

        with torch.no_grad():
            y_pred_orig = model(x).argmax(dim=1)
            y_pred_adv  = model(best_adv).argmax(dim=1)

        delta = best_adv - x
        return AttackResult(
            x_orig=x.detach(),
            x_adv=best_adv,
            y_true=y,
            y_pred_orig=y_pred_orig,
            y_pred_adv=y_pred_adv,
            perturbation_norm=self.l2_norm(delta),
            success=(y_pred_adv != y).any().item(),
            attack_name="CW",
            epsilon=self.config.epsilon,
            iterations=self.n_steps * self.binary_search_steps,
        )