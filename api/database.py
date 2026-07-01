"""
[Partner A]
SQLModel / SQLite database models and shared helpers.
Tables: AttackResult, DefenseResult, BenchmarkJob, Report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select, func

# ---------------------------------------------------------------------------
# Engine – single SQLite file shared by all routes
# ---------------------------------------------------------------------------

DATABASE_URL = "sqlite:///./aml_framework.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
    echo=False,
)


def get_session():
    """FastAPI dependency that yields a DB session."""
    with Session(engine) as session:
        yield session


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Table: AttackResult
# ---------------------------------------------------------------------------

class AttackResult(SQLModel, table=True):
    __tablename__ = "attack_results"

    sample_id: str = Field(primary_key=True)
    attack_name: str
    modality: str
    success: bool
    adversarial_accuracy: float
    asr: float
    perturbation_norm: float
    processing_time_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Table: DefenseResult
# ---------------------------------------------------------------------------

class DefenseResult(SQLModel, table=True):
    __tablename__ = "defense_results"

    id: Optional[int] = Field(default=None, primary_key=True)
    sample_id: str = Field(index=True)
    defense_type: str
    defended_accuracy: float
    defense_success_rate: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Table: BenchmarkJob
# ---------------------------------------------------------------------------

class BenchmarkJob(SQLModel, table=True):
    __tablename__ = "benchmark_jobs"

    job_id: str = Field(primary_key=True)
    status: str = Field(default="pending")  # pending | running | completed | failed
    total_steps: int = Field(default=12)
    completed_steps: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Table: Report
# ---------------------------------------------------------------------------

class Report(SQLModel, table=True):
    __tablename__ = "reports"

    report_id: str = Field(primary_key=True)
    report_text: str
    metrics_json: str = Field(default="{}")  # JSON-serialised dict[str, float]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def metrics_dict(self) -> dict:
        try:
            return json.loads(self.metrics_json)
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Helper: row counts across all four tables
# ---------------------------------------------------------------------------

def get_row_counts() -> dict[str, int]:
    """Return row counts for all four managed tables."""
    with Session(engine) as session:
        return {
            "attack_results": session.exec(select(func.count()).select_from(AttackResult)).one(),
            "defense_results": session.exec(select(func.count()).select_from(DefenseResult)).one(),
            "benchmark_jobs": session.exec(select(func.count()).select_from(BenchmarkJob)).one(),
            "reports": session.exec(select(func.count()).select_from(Report)).one(),
        }