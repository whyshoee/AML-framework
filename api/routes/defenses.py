"""
[Partner B]
Defense routes for the AML FinTech Security Framework.

Endpoints:
  POST  /defenses/apply             – apply one defense to a stored sample
  GET   /defenses/compare           – apply all 4 defenses and compare metrics
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from api.database import AttackResult, DefenseResult, get_session
from api.schemas import (
    DefenseRequest,
    DefenseResponse,
    DefenseType,
    ErrorResponse,
)
from api.ttl_cache import cache_get, cache_set

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Defenses"])


# ---------------------------------------------------------------------------
# Simulated defense computation
# ---------------------------------------------------------------------------

def _simulate_defense(defense_type: DefenseType, sample_id: str) -> dict:
    """
    Simulate defense metrics.

    Each defense type has characteristic effectiveness bands grounded in the
    AML literature:
      - adversarial_training:     high defended_accuracy, high success_rate
      - autoencoder_denoising:    moderate on both
      - input_preprocessing:      moderate accuracy, lower success_rate
      - defensive_distillation:   high accuracy for gradient-based attacks
    """
    rng = random.Random(f"{defense_type.value}:{sample_id}")

    bands: dict[DefenseType, tuple[float, float, float, float]] = {
        DefenseType.adversarial_training:   (0.75, 0.92, 0.80, 0.95),
        DefenseType.autoencoder_denoising:  (0.60, 0.80, 0.65, 0.85),
        DefenseType.input_preprocessing:    (0.55, 0.75, 0.50, 0.75),
        DefenseType.defensive_distillation: (0.70, 0.90, 0.72, 0.90),
    }
    acc_lo, acc_hi, sr_lo, sr_hi = bands[defense_type]

    logger.info(
        "Simulating defense '%s' for sample '%s'",
        defense_type.value,
        sample_id,
    )

    return {
        "defended_accuracy": round(rng.uniform(acc_lo, acc_hi), 4),
        "defense_success_rate": round(rng.uniform(sr_lo, sr_hi), 4),
    }


# ---------------------------------------------------------------------------
# POST /defenses/apply
# ---------------------------------------------------------------------------

@router.post(
    "/apply",
    response_model=DefenseResponse,
    summary="Apply a single defense to a stored adversarial sample",
)
async def apply_defense(
    body: DefenseRequest,
    session: Session = Depends(get_session),
):
    # Verify the sample exists
    record: Optional[AttackResult] = session.get(AttackResult, body.sample_id)
    if record is None:
        logger.warning(
            "apply_defense: sample not found",
            extra={"sample_id": body.sample_id},
        )
        return JSONResponse(
            status_code=404,
            content=ErrorResponse.now(
                error="Not Found",
                detail=f"No attack result with sample_id '{body.sample_id}'",
            ).model_dump(),
        )

    metrics = _simulate_defense(body.defense_type, body.sample_id)

    # Persist result
    defense_record = DefenseResult(
        sample_id=body.sample_id,
        defense_type=body.defense_type.value,
        defended_accuracy=metrics["defended_accuracy"],
        defense_success_rate=metrics["defense_success_rate"],
    )
    session.add(defense_record)
    session.commit()

    logger.info(
        "apply_defense completed",
        extra={"sample_id": body.sample_id, "defense": body.defense_type.value},
    )

    return DefenseResponse(
        defense_type=body.defense_type,
        defended_accuracy=metrics["defended_accuracy"],
        defense_success_rate=metrics["defense_success_rate"],
    )


# ---------------------------------------------------------------------------
# GET /defenses/compare?sample_id=
# ---------------------------------------------------------------------------

@router.get(
    "/compare",
    summary="Compare all 4 defense types for a given sample",
)
async def compare_defenses(
    sample_id: str = Query(..., min_length=1, description="sample_id from an attack run"),
    session: Session = Depends(get_session),
):
    cache_key = f"defense_compare:{sample_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    # Verify the sample exists
    record: Optional[AttackResult] = session.get(AttackResult, sample_id)
    if record is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse.now(
                error="Not Found",
                detail=f"No attack result with sample_id '{sample_id}'",
            ).model_dump(),
        )

    results = []
    for defense_type in DefenseType:
        metrics = _simulate_defense(defense_type, sample_id)

        # Persist each comparison result so history is preserved
        defense_record = DefenseResult(
            sample_id=sample_id,
            defense_type=defense_type.value,
            defended_accuracy=metrics["defended_accuracy"],
            defense_success_rate=metrics["defense_success_rate"],
        )
        session.add(defense_record)

        results.append(
            DefenseResponse(
                defense_type=defense_type,
                defended_accuracy=metrics["defended_accuracy"],
                defense_success_rate=metrics["defense_success_rate"],
            )
        )

    session.commit()

    response = {
        "sample_id": sample_id,
        "comparisons": [r.model_dump() for r in results],
    }
    cache_set(cache_key, response)
    return response