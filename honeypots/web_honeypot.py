"""
web_honeypot.py — [Partner B]

Production-grade Flask honeypot for the AML FinTech Security Framework.
Serves on port 5001 (default). After the iptables redirect added by the
upgraded setup_ec2.sh, port 80 traffic also arrives here — dramatically
increasing volume since port 80 is one of the most-scanned ports on the internet.

WHAT CHANGED IN THIS VERSION:
  1. Catch-all now includes HEAD, PATCH, and OPTIONS methods.
     - HEAD: used by web crawlers and availability monitors.
     - PATCH: used by REST API fuzzers (common in cloud attack tools).
     - OPTIONS: CORS preflight probes used by web scanners.
  2. Three explicit high-value decoy routes added on top of the catch-all:
     - /wp-login.php        → attracts WordPress brute-force bots
     - /.env                → attracts credential harvesters (returns fake creds)
     - /actuator/health     → attracts Spring Boot exploit scanners
     Adding explicit routes means these paths appear in the 'endpoint' feature
     with a consistent label rather than as catch-all '/<path>' strings, which
     gives the DistilBERT classifier cleaner training signal.
  3. No other changes — existing logic, feature extraction, and storage are
     unchanged to preserve data schema compatibility.

PORT NOTE:
  This Flask app still binds to 5001. Port 80 traffic arrives here via an
  iptables PREROUTING REDIRECT rule added in setup_ec2.sh:
    iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 5001
  Flask/gunicorn cannot distinguish redirected port-80 traffic from direct
  port-5001 traffic at the socket level — both appear as source:5001 after
  the REDIRECT. That is fine: what matters is logging the payload and path.

SECURITY NOTE:
  This script only ever reads and logs request data. It never executes,
  evaluates, or deserializes attacker-supplied content (no eval, no exec,
  no pickle.loads, no Jinja2 template rendering of request data).
  Keep it that way if you modify this file.

DB CONTRACT:
  database.py's get_session() is a zero-argument generator used as:
    with get_session() as session: ...
  NOT get_session(db_path). Both honeypot scripts use this contract.
"""

import argparse
import fcntl
import json
import logging
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request

# ── Make database.py importable regardless of WorkingDirectory ───────────── #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from database import HoneypotEvent, get_session
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────── #
# Constants
# ─────────────────────────────────────────────────────────────────────────── #
MAX_PAYLOAD_CHARS = 3000

SQL_KEYWORDS       = ["select", "union", "drop", "insert", "--", ";--"]
XSS_PATTERNS       = ["<script", "javascript:", "onerror=", "onload=", "alert("]
PATH_TRAVERSAL     = ["../", "%2e%2e", "/etc/passwd"]

# Base64 run of ≥20 chars — flags encoded payloads without over-flagging tokens
BASE64_RE = re.compile(r"(?:[A-Za-z0-9+/]{20,}={0,2})")

# Routes that look like legitimate FinTech API calls → label=0 IF no attack scores
VALID_FINTECH_ROUTES = {
    ("POST", "/api/v1/payment/transfer"),
    ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/kyc/verify"),
    ("GET",  "/api/v1/account/balance"),
    ("GET",  "/health"),
}

PROGRESS_LOG_INTERVAL = 500


# ─────────────────────────────────────────────────────────────────────────── #
# Logging
# ─────────────────────────────────────────────────────────────────────────── #
def setup_logging() -> logging.Logger:
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("web_honeypot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_h = logging.FileHandler(log_dir / "web_honeypot.log")
        file_h.setFormatter(fmt)
        stream_h = logging.StreamHandler(sys.stdout)
        stream_h.setFormatter(fmt)
        logger.addHandler(file_h)
        logger.addHandler(stream_h)
    return logger


logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────── #
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────── #
def json_max_depth(obj, depth: int = 0) -> int:
    """Recursive max nesting depth of a parsed JSON structure."""
    if isinstance(obj, dict) and obj:
        return max(json_max_depth(v, depth + 1) for v in obj.values())
    if isinstance(obj, list) and obj:
        return max(json_max_depth(v, depth + 1) for v in obj)
    return depth


def compute_json_depth(raw_body: str) -> int:
    try:
        return json_max_depth(json.loads(raw_body))
    except (json.JSONDecodeError, TypeError):
        return 0


def sql_score(text: str) -> int:
    lo = text.lower()
    return sum(lo.count(kw) for kw in SQL_KEYWORDS)


def xss_score(text: str) -> int:
    lo = text.lower()
    return sum(lo.count(p) for p in XSS_PATTERNS)


def path_traversal_score(text: str) -> int:
    lo = text.lower()
    return sum(lo.count(p) for p in PATH_TRAVERSAL)


def detect_base64(text: str) -> bool:
    return bool(BASE64_RE.search(text))


# ─────────────────────────────────────────────────────────────────────────── #
# Thread-safe storage (JSONL + SQLite dual-write)
# ─────────────────────────────────────────────────────────────────────────── #
class Storage:
    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()   # in-process lock (gunicorn multi-thread)
        self.total_count = 0
        self.benign_count = 0
        self.malicious_count = 0
        self.start_time = time.time()
        self.last_progress_logged = 0
        self.db_enabled = DB_AVAILABLE

    def write(self, record: dict) -> None:
        # ── (a) JSONL append with flock + threading lock ──────────────────── #
        # flock handles concurrent gunicorn worker PROCESSES.
        # threading.Lock handles concurrent threads within one worker.
        with self._lock:
            with open(self.jsonl_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(record) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            self.total_count += 1
            if record["label"] == 0:
                self.benign_count += 1
            else:
                self.malicious_count += 1

            if self.total_count - self.last_progress_logged >= PROGRESS_LOG_INTERVAL:
                self.last_progress_logged = self.total_count
                elapsed_hr = max((time.time() - self.start_time) / 3600.0, 1e-9)
                rate = self.total_count / elapsed_hr
                logger.info(
                    "Progress: %d rows | Benign: %d | Malicious: %d | Rate: %.0f req/hr",
                    self.total_count, self.benign_count, self.malicious_count, rate,
                )

        # ── (b) SQLite via SQLModel ───────────────────────────────────────── #
        if self.db_enabled:
            try:
                with get_session() as session:
                    event = HoneypotEvent(
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        source_ip=record["src_ip"],
                        source_port=0,
                        modality="text",
                        raw_payload=record["raw_payload"][:1000],
                        label="benign" if record["label"] == 0 else "attack",
                        honeypot_type="web",
                    )
                    session.add(event)
                    session.commit()
            except Exception as exc:  # noqa: BLE001
                logger.warning("DB insert failed (JSONL write succeeded): %s", exc)


# ─────────────────────────────────────────────────────────────────────────── #
# Flask app factory
# ─────────────────────────────────────────────────────────────────────────── #
def create_app(data_dir: str = "data") -> Flask:
    app = Flask(__name__)

    # ── Resolve data_dir to an absolute path ─────────────────────────────── #
    # This is the critical fix from the original README bug report:
    # relative paths silently broke when WorkingDirectory was changed to
    # honeypots/ for gunicorn. We always resolve relative to PROJECT ROOT.
    data_dir_path = Path(data_dir)
    if not data_dir_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        data_dir_path = project_root / data_dir_path

    storage = Storage(jsonl_path=data_dir_path / "raw" / "text" / "api_logs.jsonl")
    app.config["HONEYPOT_STORAGE"] = storage

    # ── Core request logger ───────────────────────────────────────────────── #
    def log_request(expected_status: int) -> None:
        """Builds the 18-feature record for the current request and stores it."""
        raw_body = ""
        try:
            raw_body = request.get_data(as_text=True) or ""
        except Exception:  # noqa: BLE001
            raw_body = ""
        raw_body_truncated = raw_body[:MAX_PAYLOAD_CHARS]

        endpoint = request.path
        method   = request.method
        qs       = request.query_string.decode("utf-8", errors="replace")

        combined = f"{endpoint} {qs} {raw_body_truncated}"

        sql   = sql_score(combined)
        xss   = xss_score(combined)
        path  = path_traversal_score(combined)
        b64   = detect_base64(raw_body_truncated)
        depth = compute_json_depth(raw_body)

        is_valid = (method, endpoint) in VALID_FINTECH_ROUTES
        label = 0 if (sql == 0 and xss == 0 and path == 0 and is_valid) else 1

        record = {
            "request_id":           str(uuid.uuid4()),
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "src_ip":               request.headers.get("X-Forwarded-For",
                                        request.remote_addr or ""),
            "endpoint":             endpoint,
            "method":               method,
            "user_agent":           request.headers.get("User-Agent", ""),
            "content_type":         request.headers.get("Content-Type", ""),
            "raw_payload":          raw_body_truncated,
            "payload_length":       len(raw_body),
            "query_string":         qs,
            "referer":              request.headers.get("Referer", ""),
            "accept_language":      request.headers.get("Accept-Language", ""),
            "json_depth":           depth,
            "sql_injection_score":  sql,
            "xss_score":            xss,
            "path_traversal_score": path,
            "base64_detected":      b64,
            "label":                label,
        }
        storage.write(record)

    # ── Decoy response headers ────────────────────────────────────────────── #
    @app.after_request
    def add_decoy_headers(response):
        # Make the honeypot look like a real FinTech production server.
        response.headers["Server"]            = "nginx/1.18.0"
        response.headers["X-Powered-By"]      = "Express"
        response.headers["X-Frame-Options"]   = "SAMEORIGIN"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    # ─────────────────────────────────────────────────────────────────────── #
    # Fake FinTech API endpoints (labelled benign IF payload is clean)
    # ─────────────────────────────────────────────────────────────────────── #
    @app.route("/api/v1/payment/transfer", methods=["POST"])
    def payment_transfer():
        log_request(401)
        return jsonify({"error": "unauthorized", "code": "AUTH_REQUIRED"}), 401

    @app.route("/api/v1/auth/login", methods=["POST"])
    def auth_login():
        log_request(401)
        return jsonify({"error": "unauthorized", "code": "INVALID_CREDENTIALS"}), 401

    @app.route("/api/v1/kyc/verify", methods=["POST"])
    def kyc_verify():
        log_request(401)
        return jsonify({"error": "unauthorized", "code": "KYC_TOKEN_REQUIRED"}), 401

    @app.route("/api/v1/account/balance", methods=["GET"])
    def account_balance():
        log_request(403)
        return jsonify({"error": "forbidden", "code": "INSUFFICIENT_PERMISSIONS"}), 403

    @app.route("/health", methods=["GET"])
    def health():
        log_request(200)
        return jsonify({"status": "ok", "version": "1.4.2"}), 200

    # ─────────────────────────────────────────────────────────────────────── #
    # High-value explicit decoy routes (NEW)
    # Defined before the catch-all so Flask matches these first.
    # Giving them explicit routes means the 'endpoint' feature in the JSONL
    # is "/wp-login.php" etc. rather than "/<path>", which gives DistilBERT
    # cleaner signal on a very common and distinctive attack class.
    # ─────────────────────────────────────────────────────────────────────── #

    @app.route("/wp-login.php", methods=["GET", "POST"])
    @app.route("/wp-admin/", methods=["GET", "POST"])
    @app.route("/wordpress/wp-login.php", methods=["GET", "POST"])
    def wp_login():
        """
        Attracts WordPress credential-stuffing bots.
        Returns a minimal fake WP login HTML — enough to keep bots probing.
        The HTML content is entirely fabricated and contains no real data.
        """
        log_request(200)
        fake_wp = (
            "<html><head><title>Log In &lsaquo; Honeypot Site — WordPress</title></head>"
            "<body><form method='post'>"
            "<input type='text' name='log' placeholder='Username'/>"
            "<input type='password' name='pwd' placeholder='Password'/>"
            "<input type='submit' value='Log In'/>"
            "</form></body></html>"
        )
        return fake_wp, 200, {"Content-Type": "text/html"}

    @app.route("/.env", methods=["GET"])
    def dotenv():
        """
        Attracts credential harvesters scanning for exposed .env files.
        Returns 100% fake credentials — no real secrets, no real keys.
        """
        log_request(200)
        fake_env = (
            "APP_NAME=FinTechPlatform\n"
            "APP_ENV=production\n"
            "APP_KEY=base64:FAKEKEYDONOTUSE1234567890ABCDEFGH=\n"
            "DB_CONNECTION=mysql\n"
            "DB_HOST=127.0.0.1\n"
            "DB_PORT=3306\n"
            "DB_DATABASE=fintech_prod\n"
            "DB_USERNAME=fintech_user\n"
            "DB_PASSWORD=FAKEPASSWORD_NOT_REAL\n"
            "STRIPE_SECRET=sk_live_FAKEKEYDONOTUSE\n"
        )
        return fake_env, 200, {"Content-Type": "text/plain"}

    @app.route("/actuator/health", methods=["GET"])
    @app.route("/actuator/env", methods=["GET"])
    @app.route("/actuator/beans", methods=["GET"])
    def actuator():
        """
        Attracts Spring Boot / Java application exploit scanners.
        These probe /actuator/* looking for exposed management endpoints.
        """
        log_request(200)
        return jsonify({"status": "UP", "components": {"db": {"status": "UP"}}}), 200

    # ─────────────────────────────────────────────────────────────────────── #
    # Catch-all route — this drives most of the 50k-row volume
    # ─────────────────────────────────────────────────────────────────────── #
    # Includes HEAD (crawlers/monitors), PATCH (REST fuzzers),
    # OPTIONS (CORS preflight probes) in addition to the original four.
    _CATCHALL_METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH", "OPTIONS"]

    @app.route("/<path:path>", methods=_CATCHALL_METHODS)
    def catch_all(path):
        log_request(404)
        return jsonify({"error": "not found"}), 404

    @app.route("/", methods=_CATCHALL_METHODS)
    def root():
        # Flask's <path:path> converter requires ≥1 path segment, so "/" needs
        # its own explicit handler.
        log_request(404)
        return jsonify({"error": "not found"}), 404

    return app


# ─────────────────────────────────────────────────────────────────────────── #
# Module-level app object — REQUIRED for `gunicorn web_honeypot:app`
# ─────────────────────────────────────────────────────────────────────────── #
app = create_app(data_dir="data")


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point  (used for manual --dev runs, NOT by the systemd service)
# ─────────────────────────────────────────────────────────────────────────── #
def main() -> None:
    parser = argparse.ArgumentParser(description="AML FinTech Web Honeypot")
    parser.add_argument("--port",        type=int, default=5001)
    parser.add_argument("--data-dir",    default="data")
    parser.add_argument("--db-path",     default="data/aml_fintech.db")
    parser.add_argument("--target-rows", type=int, default=50000)
    parser.add_argument(
        "--dev", action="store_true",
        help="Use Flask's dev server (single-threaded, no concurrency — dev only).",
    )
    args = parser.parse_args()

    global app
    app = create_app(data_dir=args.data_dir)

    logger.info(
        "Starting web honeypot on 0.0.0.0:%d (target: %d rows, mode=%s)",
        args.port, args.target_rows, "dev" if args.dev else "gunicorn",
    )

    if args.dev:
        app.run(host="0.0.0.0", port=args.port, debug=False)
    else:
        # The systemd service calls gunicorn DIRECTLY, bypassing this block.
        # This block is only used when you run `python web_honeypot.py`
        # (without --dev) from a terminal — useful for quick smoke-tests.
        module_name = Path(__file__).stem
        cmd = [
            "gunicorn",
            "-w", "4",
            "-b", f"0.0.0.0:{args.port}",
            f"{module_name}:app",
        ]
        logger.info("Launching gunicorn: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))


if __name__ == "__main__":
    main()