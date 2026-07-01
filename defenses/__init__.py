"""
[Partner A + Partner B]
defenses/__init__.py

Defense mechanism registry for the AML FinTech framework.
Exposes all four defense classes at package level so Phase 6
run_framework.py can import them with:

    from defenses import AutoencoderDefense, DefensiveDistillation
    from defenses import AdversarialTrainer, InputPreprocessor
"""

# Phase 5 Task 1 — Partner A
# from defenses.adversarial_training  import AdversarialTrainer
# from defenses.input_preprocessing   import InputPreprocessor

# Phase 5 Task 2 — Partner B
from defenses.autoencoder   import (
    AutoencoderDefense,
    ImageAutoencoder,
    ImageVAE,
    TabularAutoencoder,
    TextEmbeddingDenoiser,
)

# Phase 5 Task 3 — Partner B
from defenses.distillation  import (
    DefensiveDistillation,
    TemperatureKLLoss,
)

__all__ = [
    # Autoencoder defense
    "AutoencoderDefense",
    "ImageAutoencoder",
    "ImageVAE",
    "TabularAutoencoder",
    "TextEmbeddingDenoiser",
    # Distillation defense
    "DefensiveDistillation",
    "TemperatureKLLoss",
    # Uncomment when Partner A delivers:
    # "AdversarialTrainer",
    # "InputPreprocessor",
]