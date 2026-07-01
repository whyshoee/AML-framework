"""
network_honeypot.py — [Partner B]

UPGRADED: Multi-Port TCP Honeypot for Maximum Real-Traffic Collection.

WHY THE UPGRADE (root cause of low data volume):
  The original design listens only on port 9999 with an SSH banner.
  Internet-wide scanners (Shodan, Masscan, Mirai bots, DB exploit kits)
  focus almost exclusively on well-known service ports. Port 9999 is
  scanned by a tiny fraction of internet bots. The result: ~50-200
  connections/day instead of the 2,000-10,000+/day we need.

PORT STRATEGY (chosen for eu-north-1, the coldest AWS region):
  2222  — SSH alternative:  SSH brute-forcers scan 2222 just as heavily as 22.
                            Running on 2222 means no conflict with real sshd on 22.
    23  — Telnet:          Mirai and its variants STILL primarily target port 23.
                            This is consistently the highest-volume port on honeypots.
  3306  — MySQL:           DB exploit scanners (UDF injection, CVE-based) probe 3306
                            globally at millions of IPs/day.
  5432  — PostgreSQL:      Credential stuffers and CVE scanners target this heavily.
  6379  — Redis:           Unauthenticated Redis exploits (SLAVEOF, CONFIG SET dir)
                            are one of the most common cloud attack vectors.
 27017  — MongoDB:         Unauthenticated Mongo attacks remain extremely common.
  9999  — Original:        Kept for continuity with existing collected data.

EXPECTED VOLUME IMPROVEMENT (eu-north-1, based on similar public honeypots):
  Before: ~100 connections/day (port 9999 only)
  After:  ~3,000-15,000 connections/day across all 7 ports within 48 hours.
  Port 23 (Telnet) and 6379 (Redis) are typically the top two by volume.

ARCHITECTURE:
  One PortListener thread per port — each runs its own accept() loop.
  All ports share ONE ThreadPoolExecutor (max_workers=50) for connection handling.
  All ports share ONE HoneypotState (rate limiting, repeat tracking).
  All ports share ONE Storage instance (single CSV + single SQLite table).

  Adding 7 ports does NOT 7× the thread count. The executor is shared.

CSV SCHEMA NOTE:
  Added 'service_type' column after 'dst_port'.
  Existing 15-column CSV rows will load fine in pandas; back-fill with 'custom'
  if strict column parity is needed. Since you currently have very low volume,
  just clear the old file and restart with the new schema.

SECURITY NOTE:
  This script never executes, evaluates, or interprets attacker payload data.
  No eval, exec, subprocess with attacker args, pickle.loads, or template
  rendering of received bytes. Log-and-store only — do not break this invariant.
"""

import argparse
import csv
import fcntl
import logging
import math
import select
import signal
import socket
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# ── Make database.py importable regardless of WorkingDirectory ───────────── #
# The project root is two levels up from this file:
#   honeypots/network_honeypot.py  →  honeypots/  →  project_root/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from database import HoneypotEvent, get_session
    DB_AVAILABLE = True
except ImportError:
    # Degrade gracefully to CSV-only if database.py hasn't been deployed yet.
    DB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────── #
# Per-port fake service banners
# ─────────────────────────────────────────────────────────────────────────── #
# Each banner is sent immediately after TCP connection is established.
# Realistic banners attract the correct category of exploit bot and increase
# the probability that the bot sends a meaningful probe payload.
PORT_BANNERS: dict[int, bytes] = {
    2222: b"SSH-2.0-OpenSSH_7.4p1 Debian-10+deb9u7\r\n",

    # Telnet IAC (Interpret As Command) negotiation bytes.
    # Mirai bots recognise this exact sequence and switch to credential stuffing.
    23: (
        b"\xff\xfb\x01"   # IAC WILL ECHO
        b"\xff\xfb\x03"   # IAC WILL SUPPRESS-GO-AHEAD
        b"\xff\xfd\x18"   # IAC DO TERMINAL-TYPE
        b"\xff\xfd\x1f"   # IAC DO NAWS (window size)
    ),

    # Minimal MySQL Protocol 10 server greeting.
    # Binary format: [packet_len 3B][seq 1B][protocol 1B][version str\0]...
    # This is enough to make MySQL exploit scanners (credential stuffers,
    # CVE-2012-2122 bots, UDF-injection tools) identify this as a MySQL server.
    3306: (
        b"\x4a\x00\x00\x00"            # packet length = 74, sequence = 0
        b"\x0a"                         # protocol version 10
        b"8.0.26-honeypot\x00"          # server version string + null terminator
        b"\x08\x00\x00\x00"            # connection ID (little-endian uint32)
        b"\x52\x7a\x4f\x77\x4f\x6f\x53\x55"  # auth-plugin-data part 1 (8 bytes)
        b"\x00"                         # filler byte
        b"\xff\xf7"                     # capability flags low bytes
        b"\x21"                         # character set = utf8mb4
        b"\x02\x00"                     # status flags
        b"\xff\xff"                     # capability flags high bytes
        b"\x15"                         # length of auth-plugin-data = 21
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # reserved (10 bytes)
        b"\x3f\x4c\x4d\x53\x33\x33\x5f\x70\x64\x64\x34\x37\x00"  # auth data pt2
        b"mysql_native_password\x00"    # auth plugin name
    ),

    # PostgreSQL: sends NOTHING on connect — waits for the client's
    # 8-byte StartupMessage. PostgreSQL scanners know this and send it
    # immediately, giving us their probe payload to log.
    5432: b"",

    # Redis: sends NOTHING on connect. Automated Redis exploit tools
    # (that chain CONFIG SET dir / CONFIG SET dbfilename / SLAVEOF)
    # send their commands within milliseconds of TCP establishment.
    6379: b"",

    # MongoDB: sends NOTHING on connect. Mongo exploit tools send a
    # wire-protocol "hello" message immediately; we log it as payload.
    27017: b"",

    # Original custom port — keep the same SSH banner for continuity.
    9999: b"SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5\r\n",
}

PORT_SERVICE_TYPES: dict[int, str] = {
    2222:  "ssh_alt",
    23:    "telnet",
    3306:  "mysql",
    5432:  "postgres",
    6379:  "redis",
    27017: "mongodb",
    9999:  "custom",
}

DEFAULT_PORTS = [2222, 23, 3306, 5432, 6379, 27017, 9999]


# ─────────────────────────────────────────────────────────────────────────── #
# CSV schema  (16 columns — added 'service_type' vs the original 15)
# ─────────────────────────────────────────────────────────────────────────── #
CSV_FIELDNAMES = [
    "src_ip", "src_port", "dst_port", "service_type",
    "timestamp", "unix_timestamp",
    "bytes_received", "connection_duration_ms",
    "payload_entropy", "packet_size",
    "is_repeated_src", "src_ip_frequency",
    "payload_printable_ratio", "payload_hex_preview",
    "tcp_flags_estimated", "label",
]

# ─────────────────────────────────────────────────────────────────────────── #
# Tuning constants
# ─────────────────────────────────────────────────────────────────────────── #
RATE_LIMIT_WINDOW_SECONDS   = 60    # rolling window for rate-limit check
RATE_LIMIT_MAX_CONNECTIONS  = 500   # block an src_ip > 500 conns in 60 s
REPEATED_SRC_WINDOW_SECONDS = 300   # 5-min window for is_repeated_src feature
CONNECTION_TIMEOUT_SECONDS  = 5     # recv() gives up after this long
MAX_RECV_BYTES              = 4096  # cap to prevent unbounded memory use
PROGRESS_LOG_INTERVAL       = 100   # print progress every N total connections
MILESTONE_LOG_INTERVAL      = 1000  # INFO log every N rows


# ─────────────────────────────────────────────────────────────────────────── #
# Global shutdown flag — shared across all PortListener threads
# ─────────────────────────────────────────────────────────────────────────── #
_shutdown_requested = threading.Event()


def _handle_shutdown_signal(signum, frame):
    _shutdown_requested.set()


# ─────────────────────────────────────────────────────────────────────────── #
# Logging
# ─────────────────────────────────────────────────────────────────────────── #
def setup_logging() -> logging.Logger:
    """
    Sets up a logger that writes to both a file and stdout.
    Uses the project root (two levels up from this file) for the log dir,
    so the log location is independent of WorkingDirectory in systemd.
    """
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("network_honeypot")
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # avoid duplicate handlers on reload
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_h = logging.FileHandler(log_dir / "network_honeypot.log")
        file_h.setFormatter(fmt)
        stream_h = logging.StreamHandler(sys.stdout)
        stream_h.setFormatter(fmt)
        logger.addHandler(file_h)
        logger.addHandler(stream_h)

    return logger


# ─────────────────────────────────────────────────────────────────────────── #
# Shared cross-thread state
# ─────────────────────────────────────────────────────────────────────────── #
class HoneypotState:
    """
    All mutable state that is shared across all port listener threads and
    their connection-handler threads lives here, each guarded by its own lock.

    Keeping this in one place makes concurrency easy to audit:
    if you touch shared state outside this class, it's a bug.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Total confirmed rows written to CSV/DB (across ALL ports)
        self.total_count: int = 0
        self.start_time: float = time.time()
        self.last_milestone_logged: int = 0
        self.last_progress_logged: int = 0

        # Per-port connection counts (for logging, not rate-limiting)
        self.port_counts: dict[int, int] = defaultdict(int)

        # src_ip → total connection count (lifetime, for feature extraction)
        self.src_ip_frequency: dict[str, int] = defaultdict(int)

        # src_ip → deque of recent timestamps (for rate-limit + repeated-src)
        self.src_ip_timestamps: dict[str, deque] = defaultdict(deque)

        # Blocked IPs → time they were blocked
        self.blocklist: dict[str, float] = {}

    def register_connection(
        self, src_ip: str, now: float
    ) -> tuple[bool, bool, int]:
        """
        Records a new connection from src_ip at time `now`.

        Returns:
            is_blocked       — True if this IP is rate-limited; caller should drop.
            is_repeated_src  — True if this IP connected within the last 300 s.
            freq_total       — Total number of times this IP has ever connected.
        """
        with self._lock:
            # Fast path: already blocked
            if src_ip in self.blocklist:
                return True, False, self.src_ip_frequency[src_ip]

            timestamps = self.src_ip_timestamps[src_ip]
            timestamps.append(now)

            # Prune old timestamps outside the largest window we use
            cutoff = now - max(RATE_LIMIT_WINDOW_SECONDS, REPEATED_SRC_WINDOW_SECONDS)
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()

            # Count connections within the rate-limit window (60 s)
            recent_rate = sum(
                1 for t in timestamps if t >= now - RATE_LIMIT_WINDOW_SECONDS
            )
            # Any connection within the last 300 s (excluding the current one)
            is_repeated_src = any(
                t >= now - REPEATED_SRC_WINDOW_SECONDS
                for t in list(timestamps)[:-1]
            )

            self.src_ip_frequency[src_ip] += 1
            freq_total = self.src_ip_frequency[src_ip]

            if recent_rate > RATE_LIMIT_MAX_CONNECTIONS:
                self.blocklist[src_ip] = now
                return True, is_repeated_src, freq_total

            return False, is_repeated_src, freq_total

    def increment_total(self, port: int) -> int:
        """Increments total row count and per-port count. Returns new total."""
        with self._lock:
            self.total_count += 1
            self.port_counts[port] += 1
            return self.total_count


# ─────────────────────────────────────────────────────────────────────────── #
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────── #
def shannon_entropy(data: bytes) -> float:
    """Shannon entropy in bits/byte. Returns 0.0 for empty input."""
    if not data:
        return 0.0
    length = len(data)
    counts: dict[int, int] = defaultdict(int)
    for byte in data:
        counts[byte] += 1
    entropy = 0.0
    for c in counts.values():
        p = c / length
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def printable_ratio(data: bytes) -> float:
    """Fraction of bytes in the printable ASCII range [32, 126]. 0.0 if empty."""
    if not data:
        return 0.0
    count = sum(1 for b in data if 32 <= b <= 126)
    return round(count / len(data), 4)


def extract_features(
    src_ip: str,
    src_port: int,
    dst_port: int,
    service_type: str,
    payload: bytes,
    duration_ms: float,
    is_repeated_src: bool,
    src_ip_frequency: int,
) -> dict:
    """Builds the full 16-feature dict for one TCP connection."""
    now = datetime.now(timezone.utc)
    return {
        "src_ip":                 src_ip,
        "src_port":               src_port,
        "dst_port":               dst_port,
        "service_type":           service_type,          # NEW: e.g. "redis", "telnet"
        "timestamp":              now.isoformat(),
        "unix_timestamp":         now.timestamp(),
        "bytes_received":         len(payload),
        "connection_duration_ms": round(duration_ms, 3),
        "payload_entropy":        shannon_entropy(payload),
        "packet_size":            len(payload),
        "is_repeated_src":        is_repeated_src,
        "src_ip_frequency":       src_ip_frequency,
        "payload_printable_ratio": printable_ratio(payload),
        "payload_hex_preview":    payload[:32].hex(),
        "tcp_flags_estimated":    "SYN_ONLY" if len(payload) == 0 else "DATA",
        "label":                  1,   # honeypot assumption: all traffic is hostile
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Thread-safe storage (CSV + SQLite dual-write)
# ─────────────────────────────────────────────────────────────────────────── #
class Storage:
    """
    Handles all writes to disk. Thread-safe via:
      - threading.Lock()  (in-process, across worker threads)
      - fcntl.flock()     (inter-process, in case multiple processes ever co-exist)

    Writes are: (a) CSV append → always attempted first, (b) SQLite → attempted
    only if database.py was importable. A DB failure never silences the CSV write.
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db_enabled = DB_AVAILABLE

        # Write CSV header if the file is new/empty
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()

    def write(self, record: dict) -> None:
        # ── (a) CSV append ────────────────────────────────────────────────── #
        with self._lock:
            with open(self.csv_path, "a", newline="") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                    writer.writerow(record)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # ── (b) SQLite via SQLModel ───────────────────────────────────────── #
        # Short-lived session per write: safer than one shared session when
        # multiple threads write concurrently to the same SQLite file.
        if self.db_enabled:
            try:
                with get_session() as session:
                    event = HoneypotEvent(
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        source_ip=record["src_ip"],
                        source_port=record["src_port"],
                        modality="tabular",
                        raw_payload=record["payload_hex_preview"],
                        label="attack",
                        honeypot_type="network",
                    )
                    session.add(event)
                    session.commit()
            except Exception as exc:  # noqa: BLE001
                logging.getLogger("network_honeypot").warning(
                    "DB insert failed (CSV write succeeded): %s", exc
                )


# ─────────────────────────────────────────────────────────────────────────── #
# Connection handler (runs inside the shared ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────── #
def handle_connection(
    conn: socket.socket,
    addr: tuple,
    state: HoneypotState,
    storage: Storage,
    logger: logging.Logger,
    dst_port: int,
    service_type: str,
    banner: bytes,
    target_rows: int,
) -> None:
    """
    Handles a single incoming TCP connection from a scanner/bot.

    Steps:
      1. Rate-limit check — drop if src_ip is blocked.
      2. Send fake service banner — makes the port look like a real service.
      3. Recv payload with a 5-second timeout via select().
      4. Extract features and write to CSV + DB.
      5. Log progress milestones.
    """
    src_ip, src_port = addr
    start = time.time()

    try:
        is_blocked, is_repeated_src, freq_total = state.register_connection(
            src_ip, start
        )
        if is_blocked:
            logger.warning("Blocked %s — rate limit exceeded on port %d", src_ip, dst_port)
            conn.close()
            return

        # ── Send fake service banner ──────────────────────────────────────── #
        # Empty banner (b"") means the real service waits for client to speak
        # first (Redis, MongoDB, PostgreSQL). In that case we skip the send.
        if banner:
            try:
                conn.sendall(banner)
            except OSError:
                pass  # Client disconnected before we could send — still log it

        # ── Recv with timeout via select() ────────────────────────────────── #
        payload = b""
        conn.setblocking(False)
        readable, _, _ = select.select([conn], [], [], CONNECTION_TIMEOUT_SECONDS)
        if readable:
            try:
                payload = conn.recv(MAX_RECV_BYTES)
            except (BlockingIOError, OSError):
                payload = b""

        duration_ms = (time.time() - start) * 1000.0

        # ── Extract features and store ────────────────────────────────────── #
        record = extract_features(
            src_ip=src_ip,
            src_port=src_port,
            dst_port=dst_port,
            service_type=service_type,
            payload=payload,
            duration_ms=duration_ms,
            is_repeated_src=is_repeated_src,
            src_ip_frequency=freq_total,
        )
        storage.write(record)

        total = state.increment_total(dst_port)

        # ── Progress / milestone logging ──────────────────────────────────── #
        if total - state.last_milestone_logged >= MILESTONE_LOG_INTERVAL:
            state.last_milestone_logged = total
            logger.info(
                "Milestone: %d rows collected | Port breakdown: %s",
                total,
                dict(state.port_counts),
            )

        if total - state.last_progress_logged >= PROGRESS_LOG_INTERVAL:
            state.last_progress_logged = total
            elapsed_hr = max((time.time() - state.start_time) / 3600.0, 1e-9)
            rate = total / elapsed_hr
            remaining = max(target_rows - total, 0)
            eta_hr = remaining / rate if rate > 0 else float("inf")
            print(
                f"Progress: {total} rows | Rate: {rate:.0f} conn/hr | "
                f"ETA to {target_rows}: {eta_hr:.1f} hrs | "
                f"Latest: {src_ip}:{src_port} → port {dst_port} ({service_type})"
            )

    except Exception as exc:  # noqa: BLE001
        logger.error("Error handling connection from %s on port %d: %s", addr, dst_port, exc)
    finally:
        try:
            conn.close()
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────── #
# Per-port listener thread
# ─────────────────────────────────────────────────────────────────────────── #
class PortListener(threading.Thread):
    """
    One thread per honeypot port. Runs an accept() loop and submits
    each incoming connection to the shared ThreadPoolExecutor.

    The thread is daemon=True so it does not block Python from exiting
    once the main thread sets _shutdown_requested.
    """

    def __init__(
        self,
        port: int,
        state: HoneypotState,
        storage: Storage,
        executor: ThreadPoolExecutor,
        logger: logging.Logger,
        target_rows: int,
    ):
        super().__init__(daemon=True, name=f"listener-{port}")
        self.port = port
        self.service_type = PORT_SERVICE_TYPES.get(port, "custom")
        self.banner = PORT_BANNERS.get(port, b"")
        self.state = state
        self.storage = storage
        self.executor = executor
        self.logger = logger
        self.target_rows = target_rows
        self._server_sock: socket.socket | None = None

    def run(self) -> None:
        try:
            self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_sock.bind(("0.0.0.0", self.port))
            self._server_sock.listen(128)
            # settimeout(1.0) so the accept loop wakes up every second to
            # check the global _shutdown_requested flag.
            self._server_sock.settimeout(1.0)

            self.logger.info(
                "Listening on 0.0.0.0:%d  service=%s  banner=%r...",
                self.port,
                self.service_type,
                self.banner[:20] if self.banner else b"<none - waits for client>",
            )

            while not _shutdown_requested.is_set():
                try:
                    conn, addr = self._server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break  # Server socket was closed externally

                self.executor.submit(
                    handle_connection,
                    conn, addr,
                    self.state, self.storage, self.logger,
                    self.port, self.service_type, self.banner,
                    self.target_rows,
                )

        except Exception as exc:
            self.logger.error("PortListener on port %d crashed: %s", self.port, exc)
        finally:
            if self._server_sock:
                try:
                    self._server_sock.close()
                except OSError:
                    pass
            self.logger.info("PortListener on port %d stopped.", self.port)


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point
# ─────────────────────────────────────────────────────────────────────────── #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="AML FinTech Multi-Port Network Honeypot"
    )
    parser.add_argument(
        "--ports",
        default=",".join(str(p) for p in DEFAULT_PORTS),
        help="Comma-separated list of ports to listen on "
             f"(default: {','.join(str(p) for p in DEFAULT_PORTS)}). "
             "Each port in this list MUST also be open in the EC2 Security Group "
             "and in UFW on the instance.",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Data root directory (absolute or relative to project root). "
             "CSV will be written to <data-dir>/raw/tabular/honeypot_log.csv",
    )
    parser.add_argument("--db-path", default="data/aml_fintech.db")
    parser.add_argument("--target-rows", type=int, default=50000)
    args = parser.parse_args()

    # ── Resolve data dir to an absolute path ─────────────────────────────── #
    # Always relative to PROJECT ROOT (two levels up from this file),
    # regardless of the WorkingDirectory set in the systemd service unit.
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / data_dir

    # ── Parse port list ───────────────────────────────────────────────────── #
    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip().isdigit()]
    if not ports:
        print("ERROR: No valid ports specified. Exiting.")
        return

    # ── Set up shared resources ───────────────────────────────────────────── #
    logger = setup_logging()
    state = HoneypotState()
    storage = Storage(csv_path=data_dir / "raw" / "tabular" / "honeypot_log.csv")
    executor = ThreadPoolExecutor(max_workers=50)

    # ── Register signal handlers (SIGINT / SIGTERM for graceful shutdown) ─── #
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    # ── Start one listener thread per port ───────────────────────────────── #
    listeners: list[PortListener] = []
    for port in ports:
        listener = PortListener(
            port=port,
            state=state,
            storage=storage,
            executor=executor,
            logger=logger,
            target_rows=args.target_rows,
        )
        listener.start()
        listeners.append(listener)

    logger.info(
        "Multi-port honeypot started on %d ports: %s  |  target: %d rows",
        len(ports),
        ", ".join(str(p) for p in ports),
        args.target_rows,
    )

    # ── Main thread: block until shutdown is requested ───────────────────── #
    try:
        while not _shutdown_requested.is_set():
            time.sleep(1.0)
    finally:
        logger.info(
            "Shutdown requested. Final count: %d rows across ports %s.",
            state.total_count,
            dict(state.port_counts),
        )
        # Listeners are daemon threads — they will stop once executor shuts down
        executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()