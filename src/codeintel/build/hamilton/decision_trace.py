"""Decision trace utilities for Hamilton build runs.

Decision trace payloads are audit artifacts and must not drive control flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import msgspec

from codeintel.build.manifest.records import CacheEventStatus, CacheManifestEntry
from codeintel.core.serialization.msgspec_json import decode_json_bytes, encode_json_bytes

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


class DecisionTraceRecord(msgspec.Struct, frozen=True):
    """Serializable decision record for a cache event.

    cache_version represents the cache data version for the event.
    """

    index: int
    node_name: str
    target: str | None
    status: CacheEventStatus
    cache_key: str | None
    cache_version: str | None
    data_version: str | None
    cache_path: str | None
    duration_ms: float | None
    size_bytes: int | None
    recorded_at: str


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
                data_version=entry.cache_version,
                cache_path=entry.cache_path,
                duration_ms=entry.duration_ms,
                size_bytes=entry.size_bytes,
                recorded_at=entry.recorded_at.isoformat(),
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
    payload = msgspec.to_builtins(build_decision_trace(entries))
    return cast("list[DecisionTracePayload]", payload)


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
    payload = build_decision_trace(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_json_bytes(payload, indent=2, newline=True))


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
    raw = path.read_bytes()
    try:
        records = decode_json_bytes(raw, payload_type=list[DecisionTraceRecord])
    except msgspec.ValidationError as exc:
        data = decode_json_bytes(raw, payload_type=list[DecisionTracePayload])
        if not isinstance(data, list):
            msg = f"Decision trace payload must be a list, got {type(data)}"
            raise TypeError(msg) from exc
        return data
    payload = msgspec.to_builtins(records)
    if not isinstance(payload, list):
        msg = f"Decision trace payload must be a list, got {type(payload)}"
        raise TypeError(msg)
    return cast("list[DecisionTracePayload]", payload)


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
