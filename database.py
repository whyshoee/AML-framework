"""
database.py — [Both Partners]

SQLModel + SQLite schema for the AML FinTech Security Framework.

Schema relationships
---------------------
HoneypotEvent is the root record — one row per raw connection/request
captured by either honeypot (network_honeypot.py or web_honeypot.py).

ModelPrediction records what a trained model (XGBoost / DistilBERT /
ResNet-18) predicted for a given HoneypotEvent, including confidence and
timing — used in Phase 3+ once models are trained and run inference on
collected data.

AdversarialSample records one adversarial example generated from a source
HoneypotEvent (Phase 4) — which attack produced it, at what epsilon, and
whether it successfully flipped the model's prediction.

DefenseResult records the outcome of applying a defense (Phase 5) to a
specific AdversarialSample — did the defense recover the correct
prediction, and how long did it take.

SecurityReport stores the Claude-generated executive summaries (Phase 6)
along with the aggregate metrics used to produce them.

Usage
-----
Initialize the database once:
    python database.py

Use as a context manager (matches the honeypot scripts' usage):
    from database import get_session, HoneypotEvent
    with get_session() as session:
        session.add(HoneypotEvent(...))
        session.commit()

Use as a FastAPI dependency (Phase 6):
    from database import get_session
    @app.get("/events")
    def list_events(session: Session = Depends(get_session)):
        ...
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

# --------------------------------------------------------------------------- #
# Engine configuration
# --------------------------------------------------------------------------- #
_DB_DIR = Path(__file__).resolve().parent / "data"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DB_DIR / "aml_fintech.db"

# check_same_thread=False is required because both honeypots write from
# multiple worker threads/processes (ThreadPoolExecutor in the network
# honeypot, gunicorn workers in the web honeypot). SQLite handles this
# safely as long as each write opens its own short-lived session, which
# is exactly how both honeypot scripts use get_session().
engine = create_engine(
    f"sqlite:///{_DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)


# --------------------------------------------------------------------------- #
# Table 1 — HoneypotEvent
# --------------------------------------------------------------------------- #
class HoneypotEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime
    source_ip: str
    source_port: int
    modality: str  # "tabular" | "text" | "image"
    raw_payload: str  # JSON string or hex/text preview
    label: str  # "benign" | "attack"
    honeypot_type: str  # "network" | "web"


# --------------------------------------------------------------------------- #
# Table 2 — ModelPrediction
# --------------------------------------------------------------------------- #
class ModelPrediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(foreign_key="honeypotevent.id")
    model_name: str
    prediction: str
    confidence: float
    inference_time_ms: float
    timestamp: datetime


# --------------------------------------------------------------------------- #
# Table 3 — AdversarialSample
# --------------------------------------------------------------------------- #
class AdversarialSample(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_event_id: int = Field(foreign_key="honeypotevent.id")
    attack_type: str  # "FGSM" | "PGD" | "CW" | "DeepFool"
    epsilon: float
    modality: str
    perturbed_payload_path: str
    original_prediction: str
    adversarial_prediction: str
    attack_success: bool
    timestamp: datetime


# --------------------------------------------------------------------------- #
# Table 4 — DefenseResult
# --------------------------------------------------------------------------- #
class DefenseResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    adversarial_sample_id: int = Field(foreign_key="adversarialsample.id")
    defense_type: str
    defended_prediction: str
    defense_success: bool
    processing_time_ms: float


# --------------------------------------------------------------------------- #
# Table 5 — SecurityReport
# --------------------------------------------------------------------------- #
class SecurityReport(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    generated_at: datetime
    report_text: str  # Claude-generated plain English executive report
    clean_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    unified_robustness_score: float


# --------------------------------------------------------------------------- #
# Session management
# --------------------------------------------------------------------------- #
@contextmanager
def get_session():
    """
    Yield a SQLModel Session for use as a FastAPI dependency or context manager.

    Usage (FastAPI dependency injection)
    -------------------------------------
    >>> from database import get_session
    >>> def my_route(session: Session = Depends(get_session)):
    ...     ...

    Usage (plain context manager) — this is the pattern both honeypot
    scripts use for thread/process-safe short-lived writes:
    ------------------------------
    >>> with get_session() as session:
    ...     session.add(some_object)
    ...     session.commit()
    """
    with Session(engine) as session:
        yield session


def create_all() -> None:
    """
    Create all tables defined in this module if they do not already exist.

    Call once at application startup (e.g. from main.py or a migration script).
    SQLModel delegates to SQLAlchemy's metadata.create_all() under the hood,
    so this is idempotent — safe to call on every restart.

    Example
    -------
    >>> from database import create_all
    >>> create_all()
    """
    SQLModel.metadata.create_all(engine)
    print(f"[DB] All tables created (or verified) at: {_DB_PATH}")


# ---------------------------------------------------------------------------
# Standalone entry-point — run `python database.py` to initialise the DB
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    create_all()