"""
[Partner B]
aml_fintech_framework/defenses/autoencoder.py

Denoising Autoencoder Defense — Phase 5, Task 2
================================================
Implements modality-specific autoencoders (Standard AE + VAE for Image,
Tabular AE, Text Embedding Denoiser) with:
  - MSE-based training with early stopping (patience=5)
  - 95th-percentile reconstruction-error threshold for anomaly detection
  - defend_and_classify() pipeline
  - evaluate() producing a full metrics dict
  - plot_reconstruction_comparison() saving a matplotlib figure
  - __main__ block: quick 10-epoch image AE smoke-test on synthetic KYC data
"""

from __future__ import annotations

import os
import math
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

# ---------------------------------------------------------------------------
# Modality type alias
# ---------------------------------------------------------------------------
Modality = Literal["image", "tabular", "text"]


# ===========================================================================
# 1.  IMAGE AUTOENCODER  (Standard Convolutional AE)
# ===========================================================================
class ImageAutoencoder(nn.Module):
    """
    Convolutional denoising autoencoder for 3×64×64 KYC document images.

    Encoder: 3→32→64→128 channels, two MaxPool halving steps.
    Bottleneck: spatial size 16×16, 128 channels.
    Decoder: transposed convolutions back to 64×64, Sigmoid output ∈ [0,1].
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            # Block 1: 3×64×64 → 32×32×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # spatial: 64→64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # spatial: 64→32

            # Block 2: 32×32×32 → 64×16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # spatial: 32→16

            # Bottleneck: 64×16×16 → 128×16×16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),                         # spatial stays 16×16
        )

        # ── Decoder ────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            # Upsample: 128×16×16 → 64×32×32
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Upsample: 64×32×32 → 32×64×64
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Refine & project to RGB: 32×64×64 → 3×64×64
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),                                  # output ∈ [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode; returns reconstruction in [0, 1]."""
        return self.decoder(self.encoder(x))


# ===========================================================================
# 1b. VARIATIONAL IMAGE AUTOENCODER  (VAE)
# ===========================================================================
class ImageVAE(nn.Module):
    """
    Convolutional VAE for 3×64×64 KYC images.

    Smoother latent space → smoother reconstructions → more effective removal
    of adversarial perturbations compared to a standard AE.

    Latent dimension: latent_dim (default 128).
    The encoder outputs μ and log σ² vectors; we sample z via the
    reparameterisation trick.  KL divergence is returned alongside the
    reconstruction so the caller can add β·KL to the MSE loss.
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # ── Convolutional encoder (same topology as ImageAutoencoder) ──────
        self.conv_enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        # After encoder: 128 × 16 × 16  →  flatten to 128*16*16 = 32768
        self._flat_dim = 128 * 16 * 16

        # Linear heads for μ and log σ²
        self.fc_mu     = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

        # Project latent back to spatial map
        self.fc_dec = nn.Linear(latent_dim, self._flat_dim)

        # ── Convolutional decoder (mirrors encoder) ─────────────────────
        self.conv_dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),  nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) — shape [B, latent_dim] each."""
        h = self.conv_enc(x).view(x.size(0), -1)   # flatten
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + ε·σ with ε ~ N(0,I) (training) or use μ (eval)."""
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu                                   # deterministic at inference

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent vector back to image space."""
        h = self.fc_dec(z).view(z.size(0), 128, 16, 16)
        return self.conv_dec(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (reconstruction, mu, logvar).
        Training loss = MSE(recon, x) + β · KL, where
            KL = -0.5 · Σ(1 + logvar - μ² - exp(logvar)).
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
    """Combined MSE reconstruction + β-weighted KL divergence."""
    mse = nn.functional.mse_loss(recon, target, reduction="mean")
    # KL per element, averaged over batch
    kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kl


# ===========================================================================
# 2.  TABULAR AUTOENCODER
# ===========================================================================
class TabularAutoencoder(nn.Module):
    """
    Fully-connected denoising AE for gateway traffic feature vectors.

    Default n_features=7 matches the socket-honeypot feature set used
    by the XGBoost classifier in Phase 3.

    Encoder: n→32→16→8  (ReLU activations)
    Decoder: 8→16→32→n  (linear output — regression, no final activation)
    """

    def __init__(self, n_features: int = 7) -> None:
        super().__init__()
        self.n_features = n_features

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16),         nn.ReLU(inplace=True),
            nn.Linear(16, 8),          # bottleneck — no activation
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),  nn.ReLU(inplace=True),
            nn.Linear(16, 32), nn.ReLU(inplace=True),
            nn.Linear(32, n_features),  # linear output for regression
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ===========================================================================
# 3.  TEXT EMBEDDING DENOISER
# ===========================================================================
class TextEmbeddingDenoiser(nn.Module):
    """
    Bottleneck denoiser that operates in DistilBERT's 768-d embedding space.

    Inputs/outputs are token-level embedding tensors [B, seq_len, 768];
    each position is denoised independently (shared linear weights).

    Encoder: 768 → 256 (GELU) → 64  (bottleneck)
    Decoder: 64  → 256 (GELU) → 768
    """

    def __init__(self, hidden_dim: int = 768, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(),
            nn.Linear(256, bottleneck_dim),    # compressed representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256), nn.GELU(),
            nn.Linear(256, hidden_dim),        # reconstruct full embedding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, 768]
        Projects each token embedding through the bottleneck and back.
        The linear layers naturally share weights across the sequence dimension.
        """
        return self.decoder(self.encoder(x))


# ===========================================================================
# 4.  AutoencoderDefense  (unified wrapper)
# ===========================================================================
class AutoencoderDefense:
    """
    Modality-agnostic defense wrapper for all three autoencoder types.

    Usage
    -----
    defence = AutoencoderDefense(modality="image", model_dir="models/")
    defence.train(train_loader, n_epochs=20, save_path="models/ae_image.pt")
    defence.calibrate_threshold(val_loader)
    denoised = defence.denoise(x_adv)
    denoised_x, probs = defence.defend_and_classify(x_adv, classifier, "image")
    metrics = defence.evaluate(x_clean, x_adv, classifier)
    defence.plot_reconstruction_comparison(x_clean, x_adv, denoised)
    """

    # Supported modalities → autoencoder class
    _AE_REGISTRY = {
        "image":   ImageAutoencoder,
        "image_vae": ImageVAE,       # VAE variant for the image modality
        "tabular": TabularAutoencoder,
        "text":    TextEmbeddingDenoiser,
    }

    def __init__(
        self,
        modality: str,
        model_dir: str = "models/",
        use_vae: bool = False,
        **ae_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        modality  : one of "image", "tabular", "text"
        model_dir : directory for checkpoint I/O
        use_vae   : if True and modality=="image", use ImageVAE instead of AE
        ae_kwargs : forwarded to the autoencoder constructor
                    (e.g. n_features=12 for tabular, latent_dim=64 for VAE)
        """
        self.modality  = modality
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_vae   = use_vae and (modality == "image")

        # Instantiate the correct architecture
        key = "image_vae" if self.use_vae else modality
        if key not in self._AE_REGISTRY:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Choose from: image, tabular, text."
            )
        self.model: nn.Module = self._AE_REGISTRY[key](**ae_kwargs)

        # Anomaly detection threshold (set by calibrate_threshold)
        self.threshold: Optional[float] = None

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(
            "AutoencoderDefense — modality=%s  vae=%s  device=%s",
            modality, self.use_vae, self.device,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int = 20,
        lr: float = 1e-3,
        save_path: Optional[str] = None,
        patience: int = 5,          # early-stopping patience
        vae_beta: float = 1.0,      # β weight for KL term (VAE only)
    ) -> None:
        """
        Train the autoencoder with MSELoss and Adam optimiser.

        Early stopping
        --------------
        If the validation loss does not improve for `patience` consecutive
        epochs, training halts and the best checkpoint is restored.
        When the dataloader carries a separate validation split the first
        batch of each epoch is used as a proxy validation set (lightweight).
        For a proper split pass a val_loader via the dataloader argument that
        yields validation batches after the training batches — or calibrate
        separately with calibrate_threshold().

        Parameters
        ----------
        dataloader : yields (x,) or (x, y) batches of clean training samples
        n_epochs   : maximum training epochs
        lr         : Adam learning rate
        save_path  : .pt path to save the best model; auto-named if None
        patience   : early-stopping patience (epochs without improvement)
        vae_beta   : β coefficient for the KL term when use_vae=True
        """
        if save_path is None:
            save_path = str(
                self.model_dir / f"autoencoder_{self.modality}"
                                 f"{'_vae' if self.use_vae else ''}_best.pt"
            )

        optimizer   = optim.Adam(self.model.parameters(), lr=lr)
        mse_loss_fn = nn.MSELoss()

        best_loss        = math.inf
        epochs_no_improve = 0
        best_state       = None

        self.model.train()

        for epoch in range(1, n_epochs + 1):
            epoch_loss  = 0.0
            n_batches   = 0

            for batch in dataloader:
                # Support both (x,) and (x, label) dataloaders
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                if self.use_vae:
                    recon, mu, logvar = self.model(x)
                    loss = vae_loss(recon, x, mu, logvar, beta=vae_beta)
                else:
                    recon = self.model(x)
                    loss  = mse_loss_fn(recon, x)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info("Epoch [%3d/%d]  loss=%.6f", epoch, n_epochs, avg_loss)

            # ── Early stopping ─────────────────────────────────────────
            if avg_loss < best_loss - 1e-7:          # improvement threshold
                best_loss         = avg_loss
                epochs_no_improve = 0
                best_state        = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                torch.save(best_state, save_path)
                logger.info("  ✓ New best — saved to %s", save_path)
            else:
                epochs_no_improve += 1
                logger.info(
                    "  No improvement (%d/%d patience)", epochs_no_improve, patience
                )
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping triggered at epoch %d. Best loss=%.6f",
                        epoch, best_loss,
                    )
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
        logger.info("Training complete. Best MSE loss: %.6f", best_loss)

    # ------------------------------------------------------------------
    # Anomaly threshold calibration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate_threshold(
        self,
        val_dataloader: DataLoader,
        percentile: float = 95.0,
    ) -> float:
        """
        Compute the `percentile`-th percentile of per-sample reconstruction
        errors on clean validation data and store it as self.threshold.

        Samples whose reconstruction error exceeds this threshold are flagged
        as adversarial / anomalous at inference time.

        Parameters
        ----------
        val_dataloader : DataLoader of clean samples
        percentile     : default 95 — flags the top 5 % as anomalies

        Returns
        -------
        threshold value (also stored in self.threshold)
        """
        self.model.eval()
        errors: list[float] = []

        for batch in val_dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)

            if self.use_vae:
                recon, _, _ = self.model(x)
            else:
                recon = self.model(x)

            # Per-sample MSE: mean over all non-batch dimensions
            per_sample = (recon - x).pow(2).flatten(1).mean(dim=1)
            errors.extend(per_sample.cpu().numpy().tolist())

        self.threshold = float(np.percentile(errors, percentile))
        logger.info(
            "Threshold calibrated at %.1f-th percentile = %.6f  "
            "(n_samples=%d)",
            percentile, self.threshold, len(errors),
        )
        return self.threshold

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass x through the trained autoencoder and return the reconstruction.

        Parameters
        ----------
        x : input tensor on any device

        Returns
        -------
        Denoised tensor (same shape as x, on the same device as x)
        """
        original_device = x.device
        self.model.eval()
        x_dev = x.to(self.device)

        if self.use_vae:
            recon, _, _ = self.model(x_dev)
        else:
            recon = self.model(x_dev)

        return recon.to(original_device)

    @torch.no_grad()
    def defend_and_classify(
        self,
        x: torch.Tensor,
        classifier,
        modality: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise x then run through the classifier.

        Parameters
        ----------
        x          : (possibly adversarial) input tensor
        classifier : a callable that accepts a tensor and returns logits/probs
                     OR a torch.nn.Module with a .predict_proba() method
                     (XGBoost sklearn wrapper).
        modality   : "image" | "tabular" | "text"

        Returns
        -------
        (denoised_x, class_probabilities)
        denoised_x          — reconstructed tensor, same shape as x
        class_probabilities — softmax probabilities [B, n_classes]
        """
        denoised_x = self.denoise(x)

        # ── Classifier dispatch ────────────────────────────────────────
        if isinstance(classifier, nn.Module):
            # PyTorch model (DistilBERT, ResNet-18 wrapper)
            classifier.eval()
            logits = classifier(denoised_x.to(self.device))
            probs  = torch.softmax(logits, dim=-1)

        elif hasattr(classifier, "predict_proba"):
            # scikit-learn compatible (XGBoost)
            x_np  = denoised_x.cpu().numpy()
            probs = torch.from_numpy(
                classifier.predict_proba(x_np).astype(np.float32)
            )
        else:
            raise TypeError(
                "classifier must be a nn.Module or expose predict_proba()."
            )

        return denoised_x, probs

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        classifier,
    ) -> Dict[str, float]:
        """
        Compute four key metrics comparing clean vs adversarial performance
        before and after the autoencoder defense.

        Parameters
        ----------
        x_clean     : clean input samples  [B, ...]
        x_adv       : adversarial variants  [B, ...]
                      (same labels assumed as x_clean)
        classifier  : torch.nn.Module or sklearn-compatible predictor

        Returns
        -------
        dict with keys:
          denoising_mse           — MSE(denoise(x_adv), x_clean)
          clean_acc               — accuracy of classifier on x_clean
          adv_acc_before_defense  — accuracy on x_adv (no denoising)
          adv_acc_after_defense   — accuracy on denoise(x_adv)
        """
        self.model.eval()

        def _accuracy(inputs: torch.Tensor) -> float:
            """Helper: run classifier, return top-1 accuracy vs clean labels."""
            if isinstance(classifier, nn.Module):
                classifier.eval()
                logits = classifier(inputs.to(self.device))
                preds  = logits.argmax(dim=-1).cpu()
            else:
                preds = torch.from_numpy(
                    classifier.predict(inputs.cpu().numpy())
                )
            # Ground truth labels from clean data argmax (assumes one-hot)
            # — if x_clean is already label-encoded caller should adapt.
            if isinstance(classifier, nn.Module):
                labels = classifier(x_clean.to(self.device)).argmax(dim=-1).cpu()
            else:
                labels = torch.from_numpy(
                    classifier.predict(x_clean.cpu().numpy())
                )
            return float((preds == labels).float().mean().item())

        # 1. Reconstruction quality
        x_denoised   = self.denoise(x_adv)
        denoising_mse = float(
            nn.functional.mse_loss(x_denoised, x_clean.to(x_denoised.device))
            .item()
        )

        # 2. Accuracy metrics
        clean_acc              = _accuracy(x_clean)
        adv_acc_before_defense = _accuracy(x_adv)
        adv_acc_after_defense  = _accuracy(x_denoised)

        metrics = {
            "denoising_mse":           denoising_mse,
            "clean_acc":               clean_acc,
            "adv_acc_before_defense":  adv_acc_before_defense,
            "adv_acc_after_defense":   adv_acc_after_defense,
        }

        logger.info("── Autoencoder Evaluation Results ──────────────────")
        for k, v in metrics.items():
            logger.info("  %-30s : %.4f", k, v)
        logger.info("─────────────────────────────────────────────────────")
        return metrics

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def plot_reconstruction_comparison(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        x_denoised: torch.Tensor,
        n_samples: int = 4,
        save_dir: str = "evaluation/figures",
        filename: str = "autoencoder_reconstructions.png",
    ) -> str:
        """
        Save a side-by-side matplotlib figure with three columns:
          Clean | Adversarial | Denoised

        Works for all three modalities:
          - image   : shows the actual RGB images
          - tabular : bar chart of feature values per sample
          - text    : line plot of mean embedding magnitude per position

        Parameters
        ----------
        x_clean    : clean samples   [B, ...]
        x_adv      : adversarial     [B, ...]
        x_denoised : reconstructed   [B, ...]
        n_samples  : how many rows to plot (capped to batch size)
        save_dir   : output directory (created if needed)
        filename   : output filename

        Returns
        -------
        Absolute path to the saved figure.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        out_file  = save_path / filename

        # Move everything to CPU numpy for plotting
        def _cpu(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().float().numpy()

        xc = _cpu(x_clean)
        xa = _cpu(x_adv)
        xd = _cpu(x_denoised)

        n = min(n_samples, xc.shape[0])
        fig, axes = plt.subplots(
            n, 3,
            figsize=(9, 3 * n),
            squeeze=False,
        )
        fig.suptitle(
            "Autoencoder Reconstruction Comparison\n"
            f"modality={self.modality} | n={n}",
            fontsize=12, fontweight="bold",
        )

        col_titles = ["Clean", "Adversarial", "Denoised"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10, fontweight="bold")

        for row in range(n):
            for col, arr in enumerate([xc, xa, xd]):
                ax = axes[row, col]

                if self.modality == "image":
                    # arr[row]: [C, H, W] → [H, W, C], clipped to [0,1]
                    img = np.clip(arr[row].transpose(1, 2, 0), 0.0, 1.0)
                    ax.imshow(img, interpolation="nearest")
                    ax.axis("off")

                elif self.modality == "tabular":
                    # arr[row]: [n_features]
                    feat_vals = arr[row]
                    x_pos     = np.arange(len(feat_vals))
                    ax.bar(x_pos, feat_vals, color=["steelblue", "tomato", "seagreen"][col])
                    ax.set_xlabel("Feature index", fontsize=7)
                    ax.set_ylabel("Value",         fontsize=7)
                    ax.set_xticks(x_pos)
                    ax.tick_params(labelsize=6)

                elif self.modality == "text":
                    # arr[row]: [seq_len, hidden_dim] → mean magnitude per position
                    mag = np.abs(arr[row]).mean(axis=-1)
                    ax.plot(mag, color=["royalblue", "crimson", "forestgreen"][col],
                            linewidth=1.2)
                    ax.set_xlabel("Token position",   fontsize=7)
                    ax.set_ylabel("Mean |embedding|", fontsize=7)
                    ax.tick_params(labelsize=6)

                # Annotate reconstruction error (adv/denoised vs clean)
                if col > 0:
                    err = float(np.mean((arr[row] - xc[row]) ** 2))
                    ax.set_xlabel(
                        ax.get_xlabel() + f"\nMSE vs clean: {err:.4f}",
                        fontsize=6,
                    ) if self.modality != "image" else ax.set_title(
                        f"MSE={err:.4f}", fontsize=6, pad=2
                    )

        plt.tight_layout()
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Reconstruction comparison saved → %s", out_file.resolve())
        return str(out_file.resolve())

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist model state dict to disk."""
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved → %s", path)

    def load(self, path: str) -> None:
        """Load state dict from disk (must match instantiated architecture)."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        logger.info("Model loaded ← %s", path)


# ===========================================================================
# 5.  __main__ — quick 10-epoch smoke test on synthetic KYC image data
# ===========================================================================
if __name__ == "__main__":
    """
    Quick smoke-test: train the ImageAutoencoder on a small batch of
    synthetic 3×64×64 KYC images for 10 epochs and print the final MSE.

    Run from the project root:
        python -m aml_fintech_framework.defenses.autoencoder
        # or
        python aml_fintech_framework/defenses/autoencoder.py
    """
    import random
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print(" AML FinTech — Autoencoder Smoke Test (Image / KYC)")
    print("=" * 60)

    # ── 1. Synthetic KYC dataset ───────────────────────────────────────
    N_SAMPLES    = 256
    IMG_CHANNELS = 3
    IMG_SIZE     = 64      # 64×64 pixels

    # Simulate mild Gaussian-noise corruption (mimics adversarial perturbations)
    x_clean = torch.rand(N_SAMPLES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    x_noisy = (x_clean + 0.05 * torch.randn_like(x_clean)).clamp(0.0, 1.0)

    # DataLoader of (noisy_input, clean_target) — we pass noisy as input;
    # the train() method uses x as both input and reconstruction target
    # (denoising: train on noisy, reconstruct clean via AE compression).
    # For simplicity here we train AE on the noisy samples as self-supervised.
    dataset    = TensorDataset(x_noisy)
    train_size = int(0.8 * N_SAMPLES)
    val_size   = N_SAMPLES - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    # ── 2. Instantiate defense ─────────────────────────────────────────
    defense = AutoencoderDefense(modality="image", model_dir="models/")

    # ── 3. Train for 10 epochs (patience=5 for early stopping) ────────
    print("\n[Training Standard ImageAutoencoder — 10 epochs max]")
    defense.train(
        dataloader=train_loader,
        n_epochs=10,
        lr=1e-3,
        patience=5,
        save_path="models/ae_image_smoke_test.pt",
    )

    # ── 4. Calibrate anomaly threshold on validation set ──────────────
    print("\n[Calibrating reconstruction threshold on validation data]")
    threshold = defense.calibrate_threshold(val_loader, percentile=95.0)
    print(f"  95th-percentile threshold = {threshold:.6f}")

    # ── 5. Compute final reconstruction MSE on validation set ─────────
    all_recon_errors = []
    defense.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            xb     = batch[0].to(defense.device)
            recon  = defense.model(xb)
            mse_b  = nn.functional.mse_loss(recon, xb).item()
            all_recon_errors.append(mse_b)

    final_mse = float(np.mean(all_recon_errors))
    print(f"\n  Final Validation MSE : {final_mse:.6f}")

    # ── 6. Quick denoising visualisation ──────────────────────────────
    print("\n[Generating reconstruction comparison plot]")
    x_sample  = x_noisy[:8]                        # 8 noisy samples
    x_denoised = defense.denoise(x_sample)
    fig_path  = defense.plot_reconstruction_comparison(
        x_clean=x_clean[:8],
        x_adv=x_sample,                            # treat noisy as "adversarial"
        x_denoised=x_denoised,
        n_samples=4,
        save_dir="evaluation/figures",
    )
    print(f"  Figure saved → {fig_path}")

    # ── 7. VAE variant smoke test ──────────────────────────────────────
    print("\n[Training VAE variant — 5 epochs]")
    vae_defense = AutoencoderDefense(
        modality="image",
        model_dir="models/",
        use_vae=True,
        latent_dim=64,
    )
    vae_defense.train(
        dataloader=train_loader,
        n_epochs=5,
        lr=1e-3,
        patience=3,
        vae_beta=0.5,
        save_path="models/vae_image_smoke_test.pt",
    )
    vae_threshold = vae_defense.calibrate_threshold(val_loader)
    print(f"  VAE 95th-percentile threshold = {vae_threshold:.6f}")

    print("\n" + "=" * 60)
    print(" Smoke test complete.")
    print(f"  Standard AE  MSE  : {final_mse:.6f}")
    print(f"  Standard AE  thr  : {threshold:.6f}")
    print(f"  VAE          thr  : {vae_threshold:.6f}")
    print("=" * 60)