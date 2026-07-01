"""
[Partner B]
Report routes for the AML FinTech Security Framework.

Endpoints:
  POST  /reports/generate           – generate an LLM-backed security report
  GET   /reports/latest             – most recent report
  GET   /reports/{report_id}        – retrieve report by ID

NOTE: /reports/latest must be declared BEFORE /reports/{report_id} so FastAPI
doesn't interpret "latest" as a path parameter.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from api.database import AttackResult, DefenseResult, Report, get_session
from api.schemas import (
    AttackName,
    ErrorResponse,
    Modality,
    ReportRequest,
    ReportResponse,
)
from api.ttl_cache import cache_get, cache_set

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reports"])

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Human-readable names so the report never shows a bare enum/internal string
# to a non-technical reader.
MODEL_BY_MODALITY = {
    "tabular": "XGBoost (network traffic / fraud detection)",
    "text": "DistilBERT (API payload classification)",
    "image": "ResNet-18 (KYC document verification)",
}
ATTACK_DISPLAY = {"fgsm": "FGSM", "pgd": "PGD", "deepfool": "DeepFool", "cw": "C&W"}
DEFENSE_DISPLAY = {
    "adversarial_training": "Adversarial Training",
    "autoencoder_denoising": "Autoencoder Denoising",
    "input_preprocessing": "Input Preprocessing",
    "defensive_distillation": "Defensive Distillation",
}


def _data_context() -> dict:
    """
    Pull a quick summary of what data each modality is currently using
    (a real upload vs. the built-in synthetic default), so the report can
    tell a non-technical reader what the system was actually tested on.
    """
    try:
        import api.routes.data as _data_module  # noqa: PLC0415

        out = {}
        for modality, info in _data_module._uploaded.items():
            has_upload = bool(info.get("filename"))
            rows = info.get("cleaned_rows", info.get("rows", info.get("valid_images", 0)))
            out[modality] = {
                "source": "Your uploaded dataset" if has_upload else "Built-in sample (synthetic) dataset",
                "filename": info.get("filename"),
                "rows": rows,
                "preprocessed": info.get("preprocessed", False),
            }
        return out
    except Exception:  # noqa: BLE001
        return {}


def _build_prompt(attack_rows: list, defense_rows: list, data_ctx: dict) -> str:
    """
    Build a plain-English prompt aimed at a non-technical reader (a business
    owner or compliance person, not a security engineer). Covers: which
    models were used, what data they were tested on, which attacks were
    tried and how the system held up, which defenses were tried and how
    much they helped, and an overall risk call.
    """

    def fmt_pct(x) -> str:
        return f"{x * 100:.0f}%"

    data_lines = []
    for modality in ("tabular", "text", "image"):
        ctx = data_ctx.get(modality, {})
        model = MODEL_BY_MODALITY.get(modality, modality)
        src = ctx.get("source", "Built-in sample (synthetic) dataset")
        rows = ctx.get("rows", 0)
        data_lines.append(f"  - {modality.title()} data, handled by {model}: {src} ({rows} rows)")

    attack_lines = []
    for r in attack_rows:
        attack_lines.append(
            f"  - {ATTACK_DISPLAY.get(r['attack_name'], r['attack_name'])} attack on "
            f"{r['modality']} data: succeeded against the model "
            f"{fmt_pct(r['asr'])} of the time (model still got it right "
            f"{fmt_pct(r['adversarial_accuracy'])} of the time)."
        )
    if not attack_lines:
        attack_lines = ["  - No attacks have been run yet."]

    defense_lines = []
    for r in defense_rows:
        defense_lines.append(
            f"  - {DEFENSE_DISPLAY.get(r['defense_type'], r['defense_type'])}: "
            f"restored correct behavior {fmt_pct(r['defended_accuracy'])} of the time "
            f"({fmt_pct(r['defense_success_rate'])} success rate)."
        )
    if not defense_lines:
        defense_lines = ["  - No defenses have been applied yet."]

    prompt = f"""You are writing a short security report for a non-technical business owner who runs a
FinTech company. They have no machine learning or security background. Do not use jargon
like "ASR", "adversarial accuracy", "epsilon", or "robustness" without immediately
explaining it in plain words. Avoid acronyms where possible.

WHAT WAS TESTED (models and data):
{chr(10).join(data_lines)}

ATTACKS THAT WERE RUN AND WHAT HAPPENED:
{chr(10).join(attack_lines)}

DEFENSES THAT WERE TRIED AND HOW WELL THEY WORKED:
{chr(10).join(defense_lines)}

TASK: Write a short report with exactly these five sections, in plain English a small
business owner could read over coffee:

1. What We Tested — one short paragraph, in plain language, describing the systems and
   data involved (no model names without a one-line explanation of what each model does).
2. What We Found — summarize, per data type (network traffic, app messages, ID documents),
   how easily the system could be fooled, using percentages and a real-world consequence
   (e.g. "a fraudulent transaction could slip through").
3. What Helped — which protective measure(s) worked best, and a one-line recommendation in
   plain terms (e.g. "turn this on for all three systems").
4. Overall Risk Level — state exactly one of LOW / MEDIUM / HIGH / CRITICAL, and explain the
   rating in one plain-English sentence a non-technical reader can act on.
5. What To Do Next — three simple, concrete next steps, no jargon.

Keep the whole report under 350 words. Do not add sections beyond the five listed."""

    return prompt


def _deterministic_summary(attack_rows: list, defense_rows: list, data_ctx: dict) -> str:
    """
    Fallback when Ollama is unavailable. Mirrors the five-section plain-
    English structure used in _build_prompt, so the report reads the same
    regardless of whether the LLM is reachable.
    """

    def fmt_pct(x) -> str:
        return f"{x * 100:.0f}%"

    if not attack_rows:
        return (
            "1. What We Tested\nNo attacks have been run yet, so there is nothing to report. "
            "Go to the Attack Lab, run at least one attack, then generate a report.\n\n"
            "2. What We Found\nNot applicable — no test results yet.\n\n"
            "3. What Helped\nNot applicable — no defenses have been tried yet.\n\n"
            "4. Overall Risk Level: UNKNOWN\nNo data has been collected yet.\n\n"
            "5. What To Do Next\n1. Run an attack from the Attack Lab.\n"
            "2. Try a defense from the Defense Lab.\n3. Come back here and generate the report.\n\n"
            "(Note: AI writer unavailable — this summary was generated automatically from raw results.)"
        )

    avg_asr = sum(r["asr"] for r in attack_rows) / len(attack_rows)
    worst = max(attack_rows, key=lambda r: r["asr"])
    risk = (
        "CRITICAL" if avg_asr > 0.7 else
        "HIGH" if avg_asr > 0.5 else
        "MEDIUM" if avg_asr > 0.3 else
        "LOW"
    )

    data_lines = []
    for modality in ("tabular", "text", "image"):
        ctx = data_ctx.get(modality, {})
        model = MODEL_BY_MODALITY.get(modality, modality)
        src = ctx.get("source", "Built-in sample (synthetic) dataset")
        data_lines.append(f"- {modality.title()} data ({model}): tested using {src}.")

    found_lines = [
        f"- {ATTACK_DISPLAY.get(r['attack_name'], r['attack_name'])} on {r['modality']} data: "
        f"tricked the system {fmt_pct(r['asr'])} of the time."
        for r in attack_rows
    ]

    if defense_rows:
        best_defense = max(defense_rows, key=lambda r: r["defended_accuracy"])
        helped_text = (
            f"{DEFENSE_DISPLAY.get(best_defense['defense_type'], best_defense['defense_type'])} "
            f"worked best, restoring correct behavior {fmt_pct(best_defense['defended_accuracy'])} "
            f"of the time. Recommend turning this on everywhere before going live."
        )
    else:
        helped_text = "No defenses have been tried yet — try one from the Defense Lab to see what helps."

    return f"""1. What We Tested
{chr(10).join(data_lines)}

2. What We Found
{chr(10).join(found_lines)}
On average, attacks succeeded {fmt_pct(avg_asr)} of the time across everything tested. The biggest weak spot was {ATTACK_DISPLAY.get(worst['attack_name'], worst['attack_name'])} against the {worst['modality']} system, which could let fraudulent activity slip through undetected.

3. What Helped
{helped_text}

4. Overall Risk Level: {risk}
This is rated {risk} because attacks got through {fmt_pct(avg_asr)} of the time on average — {"a serious gap that needs attention before this handles real transactions" if risk in ("HIGH", "CRITICAL") else "a manageable level, but defenses are still worth strengthening"}.

5. What To Do Next
1. Turn on the best-performing defense above for every system, not just the one it was tested on.
2. Re-run the attacks after enabling defenses to confirm the risk level has dropped.
3. Check back periodically — re-run this test whenever the data or models change.

(Note: AI writer unavailable — this summary was generated automatically from raw results.)"""


# ---------------------------------------------------------------------------
# POST /reports/generate
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=ReportResponse,
    summary="Generate an LLM-powered security report",
    status_code=201,
)
async def generate_report(
    body: ReportRequest,
    session: Session = Depends(get_session),
):
    # -----------------------------------------------------------------------
    # Default to "everything the user has run" when no filter is given —
    # previously include_modalities/include_attacks were required+non-empty,
    # forcing the dashboard to pre-check every box just to submit a request.
    # -----------------------------------------------------------------------
    modality_values = (
        [m.value for m in body.include_modalities]
        if body.include_modalities
        else [m.value for m in Modality]
    )
    attack_values = (
        [a.value for a in body.include_attacks]
        if body.include_attacks
        else [a.value for a in AttackName]
    )

    stmt = (
        select(AttackResult)
        .where(AttackResult.modality.in_(modality_values))
        .where(AttackResult.attack_name.in_(attack_values))
        .order_by(AttackResult.created_at.desc())
    )
    records = session.exec(stmt).all()

    # Keep the most recent result per (attack, modality) combination. Query
    # is ordered by created_at desc, so the first row seen per key is the
    # newest one.
    seen_attack_keys: set[tuple[str, str]] = set()
    attack_rows: list[dict] = []
    for r in records:
        dedup_key = (r.attack_name, r.modality)
        if dedup_key in seen_attack_keys:
            continue
        seen_attack_keys.add(dedup_key)
        attack_rows.append({
            "attack_name": r.attack_name,
            "modality": r.modality,
            "asr": r.asr,
            "adversarial_accuracy": r.adversarial_accuracy,
        })

    defense_stmt = select(DefenseResult).order_by(DefenseResult.created_at.desc())
    seen_defense_keys: set[str] = set()
    defense_rows: list[dict] = []
    for d in session.exec(defense_stmt).all():
        if d.defense_type in seen_defense_keys:
            continue
        seen_defense_keys.add(d.defense_type)
        defense_rows.append({
            "defense_type": d.defense_type,
            "defended_accuracy": d.defended_accuracy,
            "defense_success_rate": d.defense_success_rate,
        })

    data_ctx = _data_context()

    # Flat metrics dict kept for ReportResponse.metrics / latest-report lookups
    metrics: dict[str, float] = {
        f"{r['attack_name']}__{r['modality']}__asr": r["asr"] for r in attack_rows
    }
    metrics.update({
        f"defense__{d['defense_type']}__defended_acc": d["defended_accuracy"] for d in defense_rows
    })

    # -----------------------------------------------------------------------
    # Call Ollama; fall back to deterministic summary on any failure
    # -----------------------------------------------------------------------
    report_text: str
    try:
        import ollama  # noqa: PLC0415 – import inside function to isolate optional dep
        import asyncio as _asyncio  # noqa: PLC0415

        prompt = _build_prompt(attack_rows, defense_rows, data_ctx)

        def _call_ollama() -> str:
            # NOTE: `options` configures Ollama *model* parameters (temperature,
            # num_ctx, etc.) — it has no "timeout" field. A genuinely hung
            # Ollama server will block this thread indefinitely; if that's a
            # concern, wrap this call with asyncio.wait_for(..., timeout=N)
            # at the call site below rather than relying on `options`.
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]

        # ollama.chat() is a blocking, synchronous network call. Calling it
        # directly inside this `async def` route previously froze the whole
        # event loop for the entire generation time — including the
        # /ws/benchmark-progress WebSocket and every other in-flight request.
        # Running it in a worker thread keeps the loop free.
        report_text = await _asyncio.wait_for(
            _asyncio.to_thread(_call_ollama), timeout=30.0
        )
        logger.info("Ollama report generated successfully")

    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama unavailable, using deterministic fallback: %s", exc)
        report_text = _deterministic_summary(attack_rows, defense_rows, data_ctx)

    # -----------------------------------------------------------------------
    # Persist the report
    # -----------------------------------------------------------------------
    report_id = str(uuid.uuid4())
    report = Report(
        report_id=report_id,
        report_text=report_text,
        metrics_json=json.dumps(metrics),
        generated_at=datetime.now(timezone.utc),
    )
    session.add(report)
    session.commit()

    logger.info("Report persisted", extra={"report_id": report_id})

    return ReportResponse(
        report_id=report_id,
        report_text=report_text,
        metrics=metrics,
        generated_at=report.generated_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /reports/latest  ← MUST come before /{report_id}
# ---------------------------------------------------------------------------

@router.get(
    "/latest",
    response_model=ReportResponse,
    summary="Return the most recently generated report",
)
async def get_latest_report(session: Session = Depends(get_session)):
    cache_key = "report:latest"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    stmt = select(Report).order_by(Report.generated_at.desc()).limit(1)
    report: Optional[Report] = session.exec(stmt).first()

    if report is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse.now(
                error="Not Found",
                detail="No reports exist yet. Generate one via POST /reports/generate.",
            ).model_dump(),
        )

    response = ReportResponse(
        report_id=report.report_id,
        report_text=report.report_text,
        metrics=report.metrics_dict(),
        generated_at=report.generated_at.isoformat(),
    )
    cache_set(cache_key, response)
    return response


# ---------------------------------------------------------------------------
# GET /reports/{report_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{report_id}",
    response_model=ReportResponse,
    summary="Retrieve a report by ID",
)
async def get_report(
    report_id: str,
    session: Session = Depends(get_session),
):
    cache_key = f"report:{report_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    report: Optional[Report] = session.get(Report, report_id)
    if report is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse.now(
                error="Not Found",
                detail=f"No report with report_id '{report_id}'",
            ).model_dump(),
        )

    response = ReportResponse(
        report_id=report.report_id,
        report_text=report.report_text,
        metrics=report.metrics_dict(),
        generated_at=report.generated_at.isoformat(),
    )
    cache_set(cache_key, response)
    return response