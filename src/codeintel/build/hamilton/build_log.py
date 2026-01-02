"""Build log buffer utilities for consolidated run diagnostics."""

from __future__ import annotations

import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.env import BuildEnv

@dataclass(frozen=True, slots=True)
class BuildLogContext:
    """Metadata for a single build log stream."""

    run_id: str
    repo: str
    commit: str
    dataset_root: Path
    snapshot_id: str


@dataclass(slots=True)
class BuildLogBuffer:
    """Thread-safe buffer for structured build events."""

    context: BuildLogContext
    events: list[dict[str, object]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


_BUFFER_VAR: ContextVar[BuildLogBuffer | None] = ContextVar(
    "codeintel_build_log_buffer",
    default=None,
)
_GLOBAL_BUFFER: BuildLogBuffer | None = None
_GLOBAL_LOCK = threading.Lock()


def _snapshot_id_from_commit(commit: str, *, run_id: str) -> str:
    value = commit.strip()
    if value:
        return value
    return run_id


def _get_buffer() -> BuildLogBuffer | None:
    buffer = _BUFFER_VAR.get()
    if buffer is not None:
        return buffer
    with _GLOBAL_LOCK:
        return _GLOBAL_BUFFER


def start_build_log(*, env: BuildEnv, run_id: str) -> BuildLogContext:
    """Initialize the build log buffer for a run."""
    context = BuildLogContext(
        run_id=run_id,
        repo=env.repo,
        commit=env.commit,
        dataset_root=env.paths.dataset_root_dir,
        snapshot_id=_snapshot_id_from_commit(env.commit, run_id=run_id),
    )
    buffer = BuildLogBuffer(context=context)
    _BUFFER_VAR.set(buffer)
    global _GLOBAL_BUFFER
    with _GLOBAL_LOCK:
        _GLOBAL_BUFFER = buffer
    return context


def record_build_event(event: str, **fields: object) -> None:
    """Append a structured event to the active build log buffer."""
    buffer = _get_buffer()
    if buffer is None:
        return
    payload = _event_payload(buffer.context, event, fields)
    with buffer.lock:
        buffer.events.append(payload)


def drain_build_log() -> tuple[BuildLogContext, list[dict[str, object]]] | None:
    """Return buffered events and clear the active buffer."""
    buffer = _get_buffer()
    if buffer is None:
        return None
    with buffer.lock:
        events = list(buffer.events)
        buffer.events.clear()
    _BUFFER_VAR.set(None)
    global _GLOBAL_BUFFER
    with _GLOBAL_LOCK:
        _GLOBAL_BUFFER = None
    return buffer.context, events


def build_log_path(*, context: BuildLogContext) -> Path:
    """Return the consolidated build log path for a run."""
    snapshot_id = _sanitize_snapshot_id(context.snapshot_id)
    return (
        context.dataset_root
        / snapshot_id
        / "build_logs"
        / f"build_{context.run_id}.jsonl"
    )


def _event_payload(
    context: BuildLogContext,
    event: str,
    fields: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": context.run_id,
        "repo": context.repo,
        "commit": context.commit,
    }
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = value
    return payload


def _sanitize_snapshot_id(snapshot_id: str) -> str:
    value = snapshot_id.strip()
    if not value:
        msg = "snapshot_id must be non-empty"
        raise ValueError(msg)
    if "/" in value or "\\" in value or value in {".", ".."}:
        msg = f"snapshot_id contains invalid characters: {snapshot_id!r}"
        raise ValueError(msg)
    return value


__all__ = [
    "BuildLogContext",
    "build_log_path",
    "drain_build_log",
    "record_build_event",
    "start_build_log",
]
