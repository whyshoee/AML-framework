"""
[Partner B] — attacks/adapters/__init__.py
==========================================
Modality adapter package.

Bridges raw gradient-based attacks to domain-specific input representations:
  • ImageAdapter   → pixel tensors          (ResNet-18  / KYC images)
  • TextAdapter    → JSON API strings       (DistilBERT / API payloads)
  • TabularAdapter → network feature arrays (XGBoost    / honeypot traffic)

Usage:
    from attacks.adapters import ImageAdapter, TextAdapter, TabularAdapter
"""

from attacks.adapters.image_adapter import ImageAdapter
from attacks.adapters.tabular_adapter import TabularAdapter, get_feature_bounds
from attacks.adapters.text_adapter import TextAdapter

__all__ = [
    "ImageAdapter",
    "TextAdapter",
    "TabularAdapter",
    "get_feature_bounds",
]
