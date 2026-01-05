"""Decision trace utilities for Hamilton build runs.

Decision trace payloads are audit artifacts and must not drive control flow.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import msgspec
from hamilton.caching.adapter import CachingEventType, HamiltonCacheAdapter

from codeintel.build.manifest.records import CacheEventStatus, CacheManifestEntry
from codeintel.core.serialization.msgspec_json import decode_json_bytes, encode_json_bytes

DECISION_TRACE_TARGET_NAME = "decision_trace"
DECISION_TRACE_ARTIFACT_NAME = "build_decision_trace"
DECISION_TRACE_PATH_TEMPLATE = "{build_dir}/decision_trace.json"
CACHE_LOG_KEY_TUPLE_LEN = 2

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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


def build_cache_manifest_entries(
    *,
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
    target_by_node: Mapping[str, str] | None = None,
    durations_ms: Mapping[str, float | None] | None = None,
) -> list[CacheManifestEntry]:
    """Build cache manifest entries from Hamilton cache logs.

    Parameters
    ----------
    cache_adapter
        Hamilton cache adapter used for the build run.
    run_id
        Hamilton run identifier.
    target_by_node
        Optional mapping of node name to target name.
    durations_ms
        Optional mapping of node name to execution duration in milliseconds.

    Returns
    -------
    list[CacheManifestEntry]
        Cache manifest entries derived from cache logs.
    """
    entries: list[CacheManifestEntry] = []
    logs_by_node = _safe_cache_logs(cache_adapter, run_id)
    if not logs_by_node:
        return entries
    for key, events in logs_by_node.items():
        node_name, task_id = _cache_log_key_parts(key)
        if not isinstance(events, list):
            continue
        cache_key = _peek_cache_key(cache_adapter, run_id, node_name, task_id)
        cache_version = _peek_cache_version(cache_adapter, run_id, node_name, cache_key, task_id)
        cache_path, size_bytes = _cache_artifact(cache_adapter, cache_version)
        target = target_by_node.get(node_name) if target_by_node else None
        duration_ms = durations_ms.get(node_name) if durations_ms else None
        for event in events:
            status = _cache_event_status(event)
            if status is None:
                continue
            recorded_at = _event_timestamp(event)
            entries.append(
                CacheManifestEntry(
                    run_id=run_id,
                    node_name=node_name,
                    status=status,
                    recorded_at=recorded_at,
                    cache_key=cache_key,
                    cache_version=cache_version if status != "miss" else None,
                    cache_path=cache_path if status != "miss" else None,
                    duration_ms=duration_ms,
                    size_bytes=size_bytes if status != "miss" else None,
                    target=target,
                )
            )
    entries.sort(key=lambda entry: entry.recorded_at)
    return entries


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


def _cache_log_key_parts(key: object) -> tuple[str, str | None]:
    if isinstance(key, str):
        return key, None
    if (
        isinstance(key, tuple)
        and len(key) == CACHE_LOG_KEY_TUPLE_LEN
        and all(isinstance(item, str) for item in key)
    ):
        return key[0], key[1]
    return str(key), None


def _safe_cache_logs(
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
) -> dict[object, object]:
    try:
        logs_by_node = cache_adapter.logs(run_id=run_id, level="info")
    except KeyError:
        return {}
    if not isinstance(logs_by_node, dict):
        return {}
    return logs_by_node


def _cache_event_status(event: object) -> CacheEventStatus | None:
    event_type = getattr(event, "event_type", None)
    msg = getattr(event, "msg", None)
    if event_type == CachingEventType.GET_RESULT and msg == "hit":
        return "hit"
    if event_type == CachingEventType.EXECUTE_NODE:
        return "miss"
    if event_type == CachingEventType.SET_RESULT:
        return "store"
    return None


def _event_timestamp(event: object) -> datetime:
    timestamp = getattr(event, "timestamp", None)
    if isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp, tz=UTC)
    return datetime.now(tz=UTC)


def _peek_cache_key(
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
    node_name: str,
    task_id: str | None,
) -> str | None:
    cache_key = cache_adapter.get_cache_key(
        run_id=run_id,
        node_name=node_name,
        task_id=task_id,
    )
    return cache_key if isinstance(cache_key, str) else None


def _peek_cache_version(
    cache_adapter: HamiltonCacheAdapter,
    run_id: str,
    node_name: str,
    cache_key: str | None,
    task_id: str | None,
) -> str | None:
    cache_version = cache_adapter.get_data_version(
        run_id=run_id,
        node_name=node_name,
        cache_key=cache_key,
        task_id=task_id,
    )
    return cache_version if isinstance(cache_version, str) else None


def _cache_artifact(
    cache_adapter: HamiltonCacheAdapter,
    cache_version: str | None,
) -> tuple[str | None, int | None]:
    if cache_version is None:
        return None, None
    result_store = cache_adapter.result_store
    if result_store is None:
        return None, None
    path_root = getattr(result_store, "path", None)
    if path_root is None:
        return None, None
    root = path_root if isinstance(path_root, Path) else Path(str(path_root))
    path = root / cache_version
    if not path.exists():
        return str(path), None
    try:
        return str(path), path.stat().st_size
    except OSError:
        return str(path), None


__all__ = [
    "DECISION_TRACE_ARTIFACT_NAME",
    "DECISION_TRACE_PATH_TEMPLATE",
    "DECISION_TRACE_TARGET_NAME",
    "DecisionTracePayload",
    "DecisionTraceRecord",
    "build_cache_manifest_entries",
    "build_decision_trace",
    "build_decision_trace_payload",
    "default_decision_trace_path",
    "read_decision_trace",
    "write_decision_trace",
]
