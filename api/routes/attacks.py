"""
[Partner A]
Attack routes for the AML FinTech Security Framework.

Endpoints:
  POST   /attacks/run                       – run a single attack (rate-limited)
  POST   /attacks/benchmark                 – launch 12-combination benchmark
  GET    /attacks/results/{sample_id}       – retrieve one result (TTL cached)
  GET    /attacks/history                   – list recent results (TTL cached)
  WS     /ws/benchmark-progress?job_id=     – live progress stream
  GET    /attacks/benchmark/{job_id}        – HTTP polling fallback
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session, select

from api.database import AttackResult, BenchmarkJob, engine, get_session
from api.schemas import (
    AttackName,
    AttackRequest,
    AttackResponse,
    BenchmarkJobStatus,
    ErrorResponse,
    Modality,
)
from api.ttl_cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (shared instance; main.py attaches it to the app)
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(tags=["Attacks"])

# ---------------------------------------------------------------------------
# Synthetic input generators (used when input_data is None)
# ---------------------------------------------------------------------------

def _synthetic_tabular() -> np.ndarray:
    """Random float32 array of shape (1, 30) representing network gateway traffic."""
    return np.random.rand(1, 30).astype(np.float32)


def _synthetic_text() -> str:
    """Fixed representative financial JSON API transaction string."""
    return "transaction amount: 500.00 USD"


def _synthetic_image() -> np.ndarray:
    """Random uint8 array of shape (1, 3, 224, 224) for KYC document verification."""
    return np.random.randint(0, 256, (1, 3, 224, 224), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Simulated attack execution
# ---------------------------------------------------------------------------

def _run_attack_simulation(
    modality: Modality,
    attack_name: AttackName,
    epsilon: float,
) -> dict:
    """
    Simulate an adversarial attack and return plausible metrics.

    In production this would call the real AML attack pipeline.
    Simulation produces deterministic-ish results keyed on attack + modality
    so benchmark comparisons are meaningful.
    """
    t_start = time.perf_counter()

    # Seed randomness per attack/modality pair for reproducibility inside a
    # single process while still varying across calls.
    rng = random.Random(f"{attack_name.value}-{modality.value}")

    # Higher epsilon → higher ASR, lower adversarial accuracy
    base_asr = min(0.95, 0.3 + epsilon * 0.65 + rng.uniform(-0.05, 0.05))
    adv_acc = max(0.02, 1.0 - base_asr + rng.uniform(-0.03, 0.03))
    p_norm = epsilon * rng.uniform(0.8, 1.2)

    # Simulate compute time (ms)
    sim_delay = {
        AttackName.fgsm: 0.05,
        AttackName.pgd: 0.15,
        AttackName.deepfool: 0.25,
        AttackName.cw: 0.40,
    }.get(attack_name, 0.1)
    time.sleep(sim_delay)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "success": base_asr > 0.5,
        "adversarial_accuracy": round(adv_acc, 4),
        "asr": round(base_asr, 4),
        "perturbation_norm": round(p_norm, 6),
        "processing_time_ms": round(elapsed_ms, 2),
    }


# ---------------------------------------------------------------------------
# POST /attacks/run
# ---------------------------------------------------------------------------

@router.post(
    "/run",
    response_model=AttackResponse,
    summary="Run a single adversarial attack",
)
@limiter.limit("10/minute")
async def run_attack(
    request: Request,  # slowapi requires Request as first positional param; imported directly
    body: AttackRequest,
    session: Session = Depends(get_session),
):
    """
    Execute one attack against the specified modality.
    Rate-limited to 10 requests/minute per IP.
    Returns HTTP 429 (ErrorResponse) when the limit is exceeded.
    """
    sample_id = str(uuid.uuid4())
    logger.info(
        "run_attack started",
        extra={"sample_id": sample_id, "attack": body.attack_name, "modality": body.modality},
    )

    metrics = _run_attack_simulation(body.modality, body.attack_name, body.epsilon)

    # Persist result
    record = AttackResult(
        sample_id=sample_id,
        attack_name=body.attack_name.value,
        modality=body.modality.value,
        **metrics,
    )
    session.add(record)
    session.commit()

    logger.info("run_attack completed", extra={"sample_id": sample_id})

    return AttackResponse(
        attack_name=body.attack_name,
        modality=body.modality,
        sample_id=sample_id,
        **metrics,
    )


# ---------------------------------------------------------------------------
# Background benchmark worker
# ---------------------------------------------------------------------------

ALL_ATTACKS = list(AttackName)
ALL_MODALITIES = list(Modality)


def _benchmark_worker(job_id: str) -> None:
    """
    Run all 4 attacks × 3 modalities = 12 combinations synchronously in a
    background thread (FastAPI BackgroundTasks runs in a thread pool executor).
    Updates DB after each step so clients can track progress.
    """
    logger.info("benchmark_worker started", extra={"job_id": job_id})

    with Session(engine) as session:
        job = session.get(BenchmarkJob, job_id)
        if job is None:
            logger.error("benchmark_worker: job not found", extra={"job_id": job_id})
            return

        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        session.add(job)
        session.commit()

        try:
            step = 0
            for modality in ALL_MODALITIES:
                for attack_name in ALL_ATTACKS:
                    sample_id = str(uuid.uuid4())
                    metrics = _run_attack_simulation(modality, attack_name, epsilon=0.1)

                    result = AttackResult(
                        sample_id=sample_id,
                        attack_name=attack_name.value,
                        modality=modality.value,
                        **metrics,
                    )
                    session.add(result)

                    step += 1
                    # Refresh the job object to avoid stale reads
                    session.refresh(job)
                    job.completed_steps = step
                    job.updated_at = datetime.now(timezone.utc)
                    session.add(job)
                    session.commit()

                    logger.info(
                        "benchmark step complete",
                        extra={"job_id": job_id, "step": step, "modality": modality, "attack": attack_name},
                    )

            session.refresh(job)
            job.status = "completed"
            job.updated_at = datetime.now(timezone.utc)
            session.add(job)
            session.commit()
            logger.info("benchmark_worker completed", extra={"job_id": job_id})

        except Exception as exc:  # noqa: BLE001
            logger.exception("benchmark_worker failed", extra={"job_id": job_id})
            try:
                session.refresh(job)
                job.status = "failed"
                job.updated_at = datetime.now(timezone.utc)
                session.add(job)
                session.commit()
            except Exception:
                pass
            raise exc


# ---------------------------------------------------------------------------
# POST /attacks/benchmark
# ---------------------------------------------------------------------------

@router.post(
    "/benchmark",
    summary="Launch 12-combination benchmark (BackgroundTask)",
    status_code=202,
)
async def launch_benchmark(
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    job_id = str(uuid.uuid4())
    job = BenchmarkJob(job_id=job_id, status="pending", total_steps=12, completed_steps=0)
    session.add(job)
    session.commit()

    background_tasks.add_task(_benchmark_worker, job_id)

    logger.info("benchmark job queued", extra={"job_id": job_id})
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# GET /attacks/results/{sample_id}
# ---------------------------------------------------------------------------

@router.get(
    "/results/{sample_id}",
    response_model=AttackResponse,
    summary="Retrieve a single attack result",
)
async def get_attack_result(
    sample_id: str,
    session: Session = Depends(get_session),
):
    cache_key = f"attack_result:{sample_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    record: Optional[AttackResult] = session.get(AttackResult, sample_id)
    if record is None:
        logger.warning("attack result not found", extra={"sample_id": sample_id})
        return JSONResponse(
            status_code=404,
            content=ErrorResponse.now(
                error="Not Found",
                detail=f"No attack result with sample_id '{sample_id}'",
            ).model_dump(),
        )

    response = AttackResponse(
        attack_name=AttackName(record.attack_name),
        modality=Modality(record.modality),
        success=record.success,
        adversarial_accuracy=record.adversarial_accuracy,
        asr=record.asr,
        perturbation_norm=record.perturbation_norm,
        processing_time_ms=record.processing_time_ms,
        sample_id=record.sample_id,
    )
    cache_set(cache_key, response)
    return response


# ---------------------------------------------------------------------------
# GET /attacks/history
# ---------------------------------------------------------------------------

@router.get(
    "/history",
    response_model=list[AttackResponse],
    summary="List recent attack results",
)
async def get_attack_history(
    modality: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
):
    # Clamp limit defensively (Query already validates, but belt-and-suspenders)
    limit = max(1, min(limit, 100))

    cache_key = f"attack_history:{modality}:{limit}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    stmt = select(AttackResult).order_by(AttackResult.created_at.desc()).limit(limit)
    if modality:
        # Validate against enum before hitting the DB
        try:
            mod_val = Modality(modality).value
        except ValueError:
            return JSONResponse(
                status_code=422,
                content=ErrorResponse.now(
                    error="Validation Error",
                    detail=f"Invalid modality '{modality}'. Must be one of: {[m.value for m in Modality]}",
                ).model_dump(),
            )
        stmt = stmt.where(AttackResult.modality == mod_val)

    records = session.exec(stmt).all()
    results = [
        AttackResponse(
            attack_name=AttackName(r.attack_name),
            modality=Modality(r.modality),
            success=r.success,
            adversarial_accuracy=r.adversarial_accuracy,
            asr=r.asr,
            perturbation_norm=r.perturbation_norm,
            processing_time_ms=r.processing_time_ms,
            sample_id=r.sample_id,
        )
        for r in records
    ]
    cache_set(cache_key, results)
    return results


# ---------------------------------------------------------------------------
# GET /attacks/benchmark/{job_id}  – HTTP polling fallback
# ---------------------------------------------------------------------------

@router.get(
    "/benchmark/{job_id}",
    response_model=BenchmarkJobStatus,
    summary="Poll benchmark job status (HTTP fallback)",
)
async def get_benchmark_status(
    job_id: str,
    session: Session = Depends(get_session),
):
    job: Optional[BenchmarkJob] = session.get(BenchmarkJob, job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse.now(
                error="Not Found",
                detail=f"No benchmark job with job_id '{job_id}'",
            ).model_dump(),
        )

    pct = round((job.completed_steps / job.total_steps) * 100, 1) if job.total_steps else 0.0
    return BenchmarkJobStatus(
        job_id=job.job_id,
        completed_steps=job.completed_steps,
        total_steps=job.total_steps,
        percentage=pct,
        status=job.status,
    )


# ---------------------------------------------------------------------------
# WebSocket /ws/benchmark-progress?job_id=
# NOTE: WebSocket routes are registered on the *app* object, not on the router,
#       because FastAPI routers do not support WebSocket prefix mounting cleanly
#       across all versions. The route is registered in main.py using
#       app.add_websocket_route or @app.websocket, but defined here so logic
#       stays co-located with attacks. main.py imports and calls
#       register_websocket(app).
# ---------------------------------------------------------------------------

async def benchmark_progress_ws(websocket: WebSocket, job_id: str):
    """
    WebSocket handler for live benchmark progress.
    On connect: immediately push current DB state (reconnect recovery).
    Then poll every 2 seconds until job is terminal.
    """
    await websocket.accept()
    logger.info("WS client connected", extra={"job_id": job_id})

    TERMINAL = {"completed", "failed"}

    try:
        while True:
            with Session(engine) as session:
                job: Optional[BenchmarkJob] = session.get(BenchmarkJob, job_id)

            if job is None:
                await websocket.send_json(
                    ErrorResponse.now(
                        error="Not Found",
                        detail=f"No benchmark job with job_id '{job_id}'",
                    ).model_dump()
                )
                break

            pct = round((job.completed_steps / job.total_steps) * 100, 1) if job.total_steps else 0.0
            payload = {
                "job_id": job.job_id,
                "completed_steps": job.completed_steps,
                "total_steps": job.total_steps,
                "percentage": pct,
                "status": job.status,
            }
            await websocket.send_json(payload)

            if job.status in TERMINAL:
                logger.info("WS job terminal, closing", extra={"job_id": job_id, "status": job.status})
                break

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        logger.info("WS client disconnected", extra={"job_id": job_id})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass  # Already closed


def register_websocket(app) -> None:
    """Called from main.py to attach the WebSocket route to the app."""

    @app.websocket("/ws/benchmark-progress")
    async def _ws_endpoint(websocket: WebSocket, job_id: str = Query(...)):
        await benchmark_progress_ws(websocket, job_id)