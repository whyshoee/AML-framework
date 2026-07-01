# attacks/__init__.py
# Exposes the public API of the attacks package.
from attacks.base import AMLModel, AttackConfig, AttackResult, compute_attack_success_rate

__all__ = ["AMLModel", "AttackConfig", "AttackResult", "compute_attack_success_rate"]