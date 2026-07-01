"""
[Partner B] — attacks/adapters/image_adapter.py
================================================
ImageAdapter — bridges gradient-based attacks to normalised pixel tensors.

DESIGN RATIONALE
----------------
ResNet-18 receives images normalised with ImageNet mean/std.  All gradient-
based attacks (FGSM, PGD, etc.) are applied in this *normalised* space so
that epsilon is meaningful relative to the data distribution.

The adapter manages the de-normalise → attack → re-normalise round-trip
transparently so that callers never have to think about it.

IMPROVEMENTS OVER SPEC
-----------------------
• NaN / Inf guard post-attack — falls back to clean input rather than crash.
• `clamp_after_denorm` option — fixes float-precision drift at the boundaries.
• `perturbed_to_pil` accepts batches [B,C,H,W] and returns a List[PIL.Image].
• `batch_attack` logs a live running ASR so progress is visible in the terminal.
• `compare_clean_vs_adv` helper — saves side-by-side PIL grid for the paper.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from attacks.base import AMLModel, AttackResult

logger = logging.getLogger(__name__)

# ---- ImageNet normalisation constants (must match training pipeline) -------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _denormalise(x: Tensor) -> Tensor:
    """Reverse ImageNet normalisation → [0, 1] pixel space."""
    mean = _IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    std  = _IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
    return x * std + mean


def _normalise(x: Tensor) -> Tensor:
    """Apply ImageNet normalisation to a [0, 1] pixel tensor."""
    mean = _IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    std  = _IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# ImageAdapter
# ---------------------------------------------------------------------------

class ImageAdapter:
    """
    Wraps any AMLModel attack so it operates correctly on KYC image tensors.

    Public API
    ----------
    apply_attack(attack, model, images, labels)  → AttackResult
    batch_attack(attack, model, dataloader, ...)  → List[AttackResult]
    perturbed_to_pil(x_adv)                      → List[PIL.Image]
    compare_clean_vs_adv(x_orig, x_adv, path)    → saves a side-by-side PNG
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Args:
            device: Torch device string ("cuda" / "cpu" / "mps").
                    Auto-detected if None.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        logger.info("ImageAdapter ready on device=%s", self.device)

    # ------------------------------------------------------------------
    # Core: single-batch attack
    # ------------------------------------------------------------------

    def apply_attack(
        self,
        attack: AMLModel,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        clamp_after_denorm: bool = True,
    ) -> AttackResult:
        """
        Apply an adversarial attack to a batch of normalised KYC images.

        Steps
        -----
        1. Move data to self.device.
        2. Run attack.generate() in normalised space.
        3. De-normalise → optionally clamp pixels to [0, 1] → re-normalise.
        4. Guard against NaN / Inf (falls back to clean image).
        5. Return AttackResult with x_adv in normalised space.

        Args:
            attack:             Any AMLModel subclass.
            model:              Trained ResNet-18 in eval mode.
            images:             Normalised tensor [B, 3, H, W].
            labels:             Ground-truth classes [B].
            clamp_after_denorm: Clamp de-normalised pixels to [0,1].
                                Set False only for ablation experiments.

        Returns:
            AttackResult — x_orig and x_adv are both in *normalised* space.
        """
        model.eval()
        images = images.to(self.device)
        labels = labels.to(self.device)

        try:
            # ---- Step 2: run gradient attack in normalised space ----
            result: AttackResult = attack.generate(images, labels, model=model)
            x_adv_norm = result.x_adv.to(self.device)

            # ---- Step 3: de-norm → clamp → re-norm ----
            x_adv_pixel = _denormalise(x_adv_norm)
            if clamp_after_denorm:
                x_adv_pixel = x_adv_pixel.clamp(0.0, 1.0)
            x_adv_renorm = _normalise(x_adv_pixel)

            # ---- Step 4: NaN / Inf guard ----
            if torch.isnan(x_adv_renorm).any() or torch.isinf(x_adv_renorm).any():
                logger.warning(
                    "[ImageAdapter] NaN/Inf in adversarial tensor — reverting to clean input."
                )
                x_adv_renorm = images.clone()

            # Overwrite x_adv with the sanitised version
            result.x_adv = x_adv_renorm

            logger.debug(
                "[ImageAdapter] attack=%s | success=%s | ‖δ‖₂=%.4f",
                result.attack_name, result.success, result.perturbation_norm,
            )
            return result

        except Exception as exc:
            logger.error("[ImageAdapter] apply_attack failed: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Batch attack over a DataLoader
    # ------------------------------------------------------------------

    def batch_attack(
        self,
        attack: AMLModel,
        model: nn.Module,
        dataloader: DataLoader,
        max_batches: int = 10,
    ) -> List[AttackResult]:
        """
        Attack up to `max_batches` batches from a DataLoader.

        Logs a running Attack Success Rate after each batch so you can
        monitor progress directly in the VS Code terminal.

        Returns:
            List of AttackResult, one per successfully processed batch.
        """
        results: List[AttackResult] = []
        model.eval()

        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            try:
                result = self.apply_attack(attack, model, images, labels)
                results.append(result)

                running_asr = sum(r.success for r in results) / len(results)
                logger.info(
                    "[ImageAdapter] batch %d/%d | running ASR=%.1f%%",
                    batch_idx + 1, max_batches, running_asr * 100,
                )
            except Exception as exc:
                logger.warning("[ImageAdapter] Skipping batch %d: %s", batch_idx, exc)
                continue

        total_asr = sum(r.success for r in results) / max(len(results), 1)
        logger.info(
            "[ImageAdapter] batch_attack done | %d batches | ASR=%.1f%%",
            len(results), total_asr * 100,
        )
        return results

    # ------------------------------------------------------------------
    # Tensor → PIL
    # ------------------------------------------------------------------

    def perturbed_to_pil(
        self,
        x_adv: Tensor,
        denormalise: bool = True,
    ) -> List[Image.Image]:
        """
        Convert an adversarial image tensor to a list of PIL Images.

        Args:
            x_adv:       Tensor [C,H,W] or [B,C,H,W].
            denormalise: Reverse ImageNet normalisation before converting.

        Returns:
            List of PIL.Image, one per sample in the batch.
        """
        if x_adv.dim() == 3:
            x_adv = x_adv.unsqueeze(0)   # → [1, C, H, W]

        x_adv = x_adv.detach().cpu().float()

        if denormalise:
            x_adv = _denormalise(x_adv)

        x_adv = x_adv.clamp(0.0, 1.0)
        to_pil = transforms.ToPILImage()
        return [to_pil(x_adv[i]) for i in range(x_adv.size(0))]

    # ------------------------------------------------------------------
    # Visualisation helper (saves side-by-side comparison PNG)
    # ------------------------------------------------------------------

    def compare_clean_vs_adv(
        self,
        x_orig: Tensor,
        x_adv: Tensor,
        save_path: str,
        n_samples: int = 4,
    ) -> None:
        """
        Save a side-by-side grid: [clean | adversarial] for n_samples images.
        Useful for the paper's qualitative results section.

        Args:
            x_orig:    Clean normalised tensor [B, C, H, W].
            x_adv:     Adversarial normalised tensor [B, C, H, W].
            save_path: File path for the output PNG.
            n_samples: How many pairs to show.
        """
        try:
            from torchvision.utils import make_grid
            import os

            n = min(n_samples, x_orig.size(0))
            orig_pil = _denormalise(x_orig[:n].cpu()).clamp(0, 1)
            adv_pil  = _denormalise(x_adv[:n].cpu()).clamp(0, 1)

            grid = make_grid(
                torch.cat([orig_pil, adv_pil], dim=0),
                nrow=n, padding=4, pad_value=1.0,
            )
            img = transforms.ToPILImage()(grid)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
            logger.info("[ImageAdapter] Comparison grid saved to %s", save_path)
        except Exception as exc:
            logger.warning("[ImageAdapter] compare_clean_vs_adv failed: %s", exc)