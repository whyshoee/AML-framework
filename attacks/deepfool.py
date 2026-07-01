# =============================================================================
# attacks/deepfool.py
# [Partner A]  — DeepFool (Moosavi-Dezfooli et al., 2016)
#
# Iteratively finds the nearest class hyperplane and steps across it:
#   Δr = - ( f_y(x) / ||∇f_y||^2 ) · ∇f_y     (linearised optimal direction)
# Adds a small overshoot factor so the final sample is robustly misclassified.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from attacks.base import AMLModel, AttackConfig, AttackResult


class DeepFool(AMLModel):
    """Minimal-norm untargeted attack via iterative linearisation."""

    def __init__(self, config: AttackConfig | None = None,
                 max_iter: int = 50, overshoot: float = 0.02):
        super().__init__(config or AttackConfig())
        self.max_iter = max_iter
        self.overshoot = overshoot

    def __repr__(self) -> str:
        return (f"DeepFool(max_iter={self.max_iter}, "
                f"overshoot={self.overshoot})")

    def _attack_single(self, x0: torch.Tensor,
                       model: nn.Module) -> tuple[torch.Tensor, int]:
        """
        Run DeepFool on a single sample x0 (shape: [C, H, W] or [D]).
        Returns (x_adv, iterations_used).
        """
        x = x0.unsqueeze(0).clone().detach()   # [1, ...]
        num_classes = None
        iters_used  = 0

        for i in range(self.max_iter):
            x.requires_grad_(True)
            logits = model(x)                   # [1, K]

            if num_classes is None:
                num_classes = logits.size(1)

            pred = logits.argmax(dim=1).item()
            if pred != x0.argmax().item() if x0.dim() == 1 else False:
                break  # already misclassified on first iter

            # gradient of the correct-class score
            grad_orig = torch.autograd.grad(
                logits[0, pred], x, retain_graph=True
            )[0].detach()

            # find the class whose boundary is nearest
            min_dist    = float("inf")
            best_r      = torch.zeros_like(x)

            for k in range(num_classes):
                if k == pred:
                    continue
                grad_k = torch.autograd.grad(
                    logits[0, k], x, retain_graph=True
                )[0].detach()

                w_k = (grad_k - grad_orig).flatten()
                f_k = (logits[0, k] - logits[0, pred]).item()

                dist = abs(f_k) / (w_k.norm(p=2).item() + 1e-12)
                if dist < min_dist:
                    min_dist = dist
                    # minimal perturbation to reach k-th boundary
                    r_k = (abs(f_k) / (w_k.norm(p=2) ** 2 + 1e-12)) * (
                        grad_k - grad_orig
                    )
                    best_r = r_k

            # apply perturbation with overshoot
            with torch.no_grad():
                x = x + (1 + self.overshoot) * best_r
                x = x.clamp(self.config.clip_min, self.config.clip_max)
            x = x.detach()
            iters_used = i + 1

            # stop once misclassified
            with torch.no_grad():
                if model(x).argmax(dim=1).item() != pred:
                    break

        return x.squeeze(0).detach(), iters_used

    def generate(self, x: torch.Tensor, y: torch.Tensor,
                 model: nn.Module, **kwargs) -> AttackResult:
        model.eval()
        adv_list  = []
        iters_sum = 0

        for i in range(x.size(0)):
            x_adv_i, iters = self._attack_single(x[i], model)
            adv_list.append(x_adv_i)
            iters_sum += iters

        x_adv = torch.stack(adv_list, dim=0)

        with torch.no_grad():
            y_pred_orig = model(x).argmax(dim=1)
            y_pred_adv  = model(x_adv).argmax(dim=1)

        delta = x_adv - x
        return AttackResult(
            x_orig=x.detach(),
            x_adv=x_adv,
            y_true=y,
            y_pred_orig=y_pred_orig,
            y_pred_adv=y_pred_adv,
            perturbation_norm=self.l2_norm(delta),
            success=(y_pred_adv != y).any().item(),
            attack_name="DeepFool",
            epsilon=self.config.epsilon,
            iterations=iters_sum // max(x.size(0), 1),
        )