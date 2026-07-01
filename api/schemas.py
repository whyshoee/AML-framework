"""
[Partner A]
Pydantic v2 schemas for the AML FinTech Security Framework API.
Includes enums, validated request/response models, and a shared ErrorResponse.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums (implemented as str sub-classes so they serialise cleanly to JSON)
# ---------------------------------------------------------------------------

from enum import Enum


class Modality(str, Enum):
    tabular = "tabular"
    text = "text"
    image = "image"


class AttackName(str, Enum):
    fgsm = "fgsm"
    pgd = "pgd"
    deepfool = "deepfool"
    cw = "cw"


class DefenseType(str, Enum):
    adversarial_training = "adversarial_training"
    autoencoder_denoising = "autoencoder_denoising"
    input_preprocessing = "input_preprocessing"
    defensive_distillation = "defensive_distillation"


# ---------------------------------------------------------------------------
# Shared error response (used for 404, 422, 500)
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str  # ISO-8601 string

    @classmethod
    def now(cls, error: str, detail: str) -> "ErrorResponse":
        return cls(
            error=error,
            detail=detail,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------------------------
# Attack schemas
# ---------------------------------------------------------------------------

class AttackRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "modality": "tabular",
                "attack_name": "fgsm",
                "epsilon": 0.1,
                "input_data": None,
                "label": 0,
            }
        }
    )

    modality: Modality
    attack_name: AttackName
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0, description="Perturbation magnitude in [0.0, 1.0]")
    # Optional: base64-encoded bytes or a JSON string. When None, a synthetic
    # input is generated server-side so callers can test without real data.
    input_data: Optional[str] = Field(default=None, description="Base64-encoded or JSON input; omit to use synthetic data")
    label: int = Field(default=0, ge=0, description="Ground-truth class label (non-negative integer)")


class AttackResponse(BaseModel):
    attack_name: AttackName
    modality: Modality
    success: bool
    adversarial_accuracy: float
    asr: float                  # Attack Success Rate
    perturbation_norm: float
    processing_time_ms: float
    sample_id: str


# ---------------------------------------------------------------------------
# Defense schemas
# ---------------------------------------------------------------------------

class DefenseRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sample_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "defense_type": "adversarial_training",
            }
        }
    )

    sample_id: str = Field(min_length=1, description="Non-empty sample ID returned by an attack run")
    defense_type: DefenseType


class DefenseResponse(BaseModel):
    defense_type: DefenseType
    defended_accuracy: float
    defense_success_rate: float


# ---------------------------------------------------------------------------
# Report schemas
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {}
        }
    )

    # Both filters are optional. Omit them (or send empty lists) to include
    # every modality / attack the user has actually run — this is the
    # default and recommended usage from the dashboard's single
    # "Generate report" button. Previously these were required+non-empty,
    # which forced the dashboard to pre-check every box just to make a
    # valid request.
    include_modalities: Optional[List[Modality]] = Field(
        default=None,
        description="Modalities to include. Omit to include everything that was run.",
    )
    include_attacks: Optional[List[AttackName]] = Field(
        default=None,
        description="Attacks to include. Omit to include everything that was run.",
    )


class ReportResponse(BaseModel):
    report_id: str = Field(min_length=1)
    report_text: str
    metrics: Dict[str, float]
    generated_at: str  # ISO-8601


# ---------------------------------------------------------------------------
# Pagination / list helpers
# ---------------------------------------------------------------------------

class HistoryQueryParams(BaseModel):
    """Validated query parameters for GET /attacks/history."""
    modality: Optional[Modality] = None
    limit: int = Field(default=20, ge=1, le=100)


# ---------------------------------------------------------------------------
# Benchmark job status (returned by HTTP polling fallback and WebSocket)
# ---------------------------------------------------------------------------

class BenchmarkJobStatus(BaseModel):
    job_id: str
    completed_steps: int
    total_steps: int
    percentage: float
    status: str  # pending | running | completed | failed