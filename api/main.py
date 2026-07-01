"""
[Partner A + Partner B]
AML FinTech Security Framework — FastAPI application entry point.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Architecture notes:
  - slowapi is attached to the app so its rate-limit middleware fires globally.
  - WebSocket /ws/benchmark-progress is registered directly on the app because
    FastAPI's APIRouter does not support WS mounting with prefix cleanly in all
    versions; the handler lives in routes/attacks.py (register_websocket).
  - All error responses use the shared ErrorResponse schema.
  - last_health_check is a module-level variable updated on every /health hit.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlmodel import Session, select

from api.database import (
    AttackResult,
    BenchmarkJob,
    DefenseResult,
    Report,
    create_db_and_tables,
    engine,
    get_row_counts,
)
from api.routes.attacks import limiter, register_websocket
from api.routes.attacks import router as attacks_router
from api.routes.data import router as data_router
from api.routes.models import router as models_router
from api.routes.defenses import router as defenses_router
from api.routes.reports import router as reports_router
from api.schemas import ErrorResponse

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_process_start: float = time.time()
_last_health_check: Optional[str] = None  # ISO timestamp, updated on each /health hit

# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AML FinTech Security Framework API",
    version="1.0.0",
    description=(
        "Adversarial Machine Learning Attack and Defense Framework "
        "for FinTech API Security. Supports tabular, text, and image modalities."
    ),
)

# Attach slowapi state so the rate-limit middleware can store counters
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS (development mode — all origins allowed)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(attacks_router, prefix="/api/v1/attacks")
app.include_router(defenses_router, prefix="/api/v1/defenses")
app.include_router(reports_router, prefix="/api/v1/reports")
app.include_router(data_router,    prefix="/api/v1/data")
app.include_router(models_router,  prefix="/api/v1/models")

# Register WebSocket route directly on the app (see module docstring)
register_websocket(app)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    create_db_and_tables()
    logger.info("Framework API started")


# ---------------------------------------------------------------------------
# Custom exception handlers (all return ErrorResponse JSON)
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse.now(
            error="Not Found",
            detail=str(exc.detail) if hasattr(exc, "detail") else "Resource not found",
        ).model_dump(),
    )


@app.exception_handler(422)
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc) -> JSONResponse:
    # FastAPI's automatic request validation raises RequestValidationError, which
    # is NOT an HTTPException subclass — Starlette only does int-status-code
    # lookup for HTTPException/WebSocketException, so registering on 422 alone
    # never catches it. Registering both ensures both explicitly-raised
    # HTTPException(422, ...) and FastAPI's automatic validation errors land here.
    detail = str(exc.errors()) if hasattr(exc, "errors") else str(exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse.now(
            error="Validation Error",
            detail=detail,
        ).model_dump(),
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc) -> JSONResponse:
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse.now(
            error="Internal Server Error",
            detail="An unexpected error occurred. Check server logs for details.",
        ).model_dump(),
    )


# Override slowapi's default 429 response to use ErrorResponse
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=ErrorResponse.now(
            error="Too Many Requests",
            detail=f"Rate limit exceeded: {exc.detail}",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["Meta"])
async def root():
    return {"status": "running", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
async def health():
    """
    Lightweight health check.
    Checks DB connectivity (count query on attack_results).
    models_loaded is always True in this simulation framework; swap in real
    model-loading checks when integrating XGBoost/DistilBERT/ResNet.
    """
    global _last_health_check

    _last_health_check = datetime.now(timezone.utc).isoformat()

    # Minimal DB probe — fail fast without heavy queries
    db_ok = "ok"
    try:
        with Session(engine) as session:
            session.exec(select(AttackResult).limit(1)).first()
    except Exception as exc:
        logger.error("Health check DB error: %s", exc)
        db_ok = "error"

    return {
        "db": db_ok,
        "models_loaded": True,  # Simulated — replace with real checks in production
    }


# ---------------------------------------------------------------------------
# Extended status
# ---------------------------------------------------------------------------

@app.get("/api/v1/status", tags=["Meta"])
async def extended_status():
    """
    Detailed operational status:
      - per-model loading flags
      - DB row counts for all four tables
      - process uptime
      - active benchmark jobs count
      - timestamp of the most recent /health call
    """
    row_counts = get_row_counts()

    # Count active (running or pending) benchmark jobs
    with Session(engine) as session:
        active_jobs: int = len(
            session.exec(
                select(BenchmarkJob).where(
                    BenchmarkJob.status.in_(["running", "pending"])
                )
            ).all()
        )

    uptime = round(time.time() - _process_start, 2)

    return {
        "models_loaded": {
            "XGBoost": True,       # Simulated — replace with real flag
            "DistilBERT": True,    # Simulated
            "ResNet-18": True,     # Simulated
        },
        "db_row_counts": row_counts,
        "uptime_seconds": uptime,
        "active_benchmark_jobs": active_jobs,
        "last_health_check": _last_health_check,
    }