"""
[Partner A]
Module-level TTL cache implementation.

Stores entries as {cache_key: (value, expiry_timestamp)}.
Entries are evicted lazily on read when time.time() > expiry.
Do NOT use lru_cache for TTL semantics – this is the canonical cache.
"""

from __future__ import annotations

import time
from typing import Any, Optional

# Module-level cache store shared across the process.
_cache: dict[str, tuple[Any, float]] = {}

DEFAULT_TTL: float = 60.0  # seconds


def cache_get(key: str) -> Optional[Any]:
    """Return cached value if present and not expired; otherwise None."""
    entry = _cache.get(key)
    if entry is None:
        return None
    value, expiry = entry
    if time.time() > expiry:
        # Evict stale entry on read
        _cache.pop(key, None)
        return None
    return value


def cache_set(key: str, value: Any, ttl: float = DEFAULT_TTL) -> None:
    """Store value in cache with the given TTL (seconds)."""
    _cache[key] = (value, time.time() + ttl)


def cache_delete(key: str) -> None:
    """Remove a key from the cache, if present."""
    _cache.pop(key, None)