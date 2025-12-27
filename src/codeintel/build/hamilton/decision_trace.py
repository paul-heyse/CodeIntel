"""Decision trace utilities for Hamilton build runs.

Decision trace payloads are audit artifacts and must not drive control flow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from codeintel.build.manifest.records import CacheEventStatus, CacheManifestEntry

DECISION_TRACE_TARGET_NAME = "decision_trace"
DECISION_TRACE_ARTIFACT_NAME = "build_decision_trace"
DECISION_TRACE_PATH_TEMPLATE = "{build_dir}/decision_trace.json"

if TYPE_CHECKING:
    from collections.abc import Sequence


class DecisionTracePayload(TypedDict):
    """Typed payload for decision trace JSON entries."""

    index: int
    node_name: str
    target: str | None
    status: CacheEventStatus
    cache_key: str | None
    cache_version: str | None
    data_version: NotRequired[str | None]
    cache_path: str | None
    duration_ms: float | None
    size_bytes: int | None
    recorded_at: str


@dataclass(frozen=True, slots=True)
class DecisionTraceRecord:
    """Serializable decision record for a cache event.

    cache_version represents the cache data version for the event.
    """

    index: int
    node_name: str
    target: str | None
    status: CacheEventStatus
    cache_key: str | None
    cache_version: str | None
    cache_path: str | None
    duration_ms: float | None
    size_bytes: int | None
    recorded_at: datetime

    def to_dict(self) -> DecisionTracePayload:
        """Return a JSON-ready dictionary.

        Returns
        -------
        dict[str, object]
            JSON-serializable mapping for this record.
        """
        return {
            "index": self.index,
            "node_name": self.node_name,
            "target": self.target,
            "status": self.status,
            "cache_key": self.cache_key,
            "cache_version": self.cache_version,
            "data_version": self.cache_version,
            "cache_path": self.cache_path,
            "duration_ms": self.duration_ms,
            "size_bytes": self.size_bytes,
            "recorded_at": self.recorded_at.isoformat(),
        }


def build_decision_trace(entries: Sequence[CacheManifestEntry]) -> list[DecisionTraceRecord]:
    """Build decision trace records from cache manifest entries.

    Returns
    -------
    list[DecisionTraceRecord]
        Ordered decision trace records for the cache events.
    """
    records: list[DecisionTraceRecord] = []
    for idx, entry in enumerate(entries):
        records.append(
            DecisionTraceRecord(
                index=idx,
                node_name=entry.node_name,
                target=entry.target,
                status=entry.status,
                cache_key=entry.cache_key,
                cache_version=entry.cache_version,
                cache_path=entry.cache_path,
                duration_ms=entry.duration_ms,
                size_bytes=entry.size_bytes,
                recorded_at=entry.recorded_at,
            )
        )
    return records


def build_decision_trace_payload(
    entries: Sequence[CacheManifestEntry],
) -> list[DecisionTracePayload]:
    """Return decision trace payload as a JSON-ready list.

    Returns
    -------
    list[dict[str, object]]
        JSON-serializable decision trace payload.
    """
    return [record.to_dict() for record in build_decision_trace(entries)]


def default_decision_trace_path(build_dir: Path) -> Path:
    """Return the default decision trace path within the build directory.

    Returns
    -------
    Path
        Filesystem path for the decision trace payload.
    """
    return build_dir / "decision_trace.json"


def write_decision_trace(path: Path, entries: Sequence[CacheManifestEntry]) -> None:
    """Write decision trace JSON to the provided path."""
    payload = build_decision_trace_payload(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_text = json.dumps(payload, indent=2)
    path.write_text(f"{payload_text}\n", encoding="utf-8")


def read_decision_trace(path: Path) -> list[DecisionTracePayload]:
    """Load decision trace JSON from disk.

    Returns
    -------
    list[dict[str, object]]
        Parsed decision trace payload.

    Raises
    ------
    TypeError
        If the payload is not a list.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"Decision trace payload must be a list, got {type(data)}"
        raise TypeError(msg)
    return cast("list[DecisionTracePayload]", data)


__all__ = [
    "DECISION_TRACE_ARTIFACT_NAME",
    "DECISION_TRACE_PATH_TEMPLATE",
    "DECISION_TRACE_TARGET_NAME",
    "DecisionTracePayload",
    "DecisionTraceRecord",
    "build_decision_trace",
    "build_decision_trace_payload",
    "default_decision_trace_path",
    "read_decision_trace",
    "write_decision_trace",
]
