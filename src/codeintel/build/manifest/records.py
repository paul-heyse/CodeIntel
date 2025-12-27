"""Cache manifest record types for build audit logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

CacheEventStatus = Literal["hit", "miss", "store"]


@dataclass(frozen=True, slots=True)
class CacheManifestEntry:
    """Record a cache event for a single Hamilton node.

    Attributes
    ----------
    run_id
        Hamilton run identifier.
    node_name
        Hamilton node name that triggered the event.
    status
        Cache event status: hit, miss, or store.
    recorded_at
        Timestamp when the event was recorded.
    cache_key
        Cache key computed for the node.
    cache_version
        Cache data version associated with the event.
    cache_path
        Resolved cache artifact path when available.
    duration_ms
        Optional duration in milliseconds for the node execution.
    size_bytes
        Optional size of the cached payload in bytes.
    target
        Optional target name inferred from node tags.
    """

    run_id: str
    node_name: str
    status: CacheEventStatus
    recorded_at: datetime
    cache_key: str | None = None
    cache_version: str | None = None
    cache_path: str | None = None
    duration_ms: float | None = None
    size_bytes: int | None = None
    target: str | None = None


__all__ = [
    "CacheEventStatus",
    "CacheManifestEntry",
]
