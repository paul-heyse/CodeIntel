"""Runtime registries for teardown telemetry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from threading import Lock
from time import monotonic

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SubprocessRecord:
    """Tracked subprocess metadata for teardown reporting."""

    pid: int
    command: str
    started_at: float
    last_seen: float
    exit_code: int | None = None
    duration_ms: float | None = None


_SUBPROCESS_LOCK = Lock()
_SUBPROCESS_REGISTRY: dict[int, SubprocessRecord] = {}
_STALE_COMPLETED_TTL_S = 300.0


def register_subprocess(*, pid: int, command: str) -> None:
    """Register a subprocess for teardown reporting.

    Parameters
    ----------
    pid
        Process identifier for the subprocess.
    command
        Command basename for the subprocess.
    """
    if pid <= 0:
        log.warning("Skipping subprocess registration for invalid pid=%s", pid)
        return
    now = monotonic()
    record = SubprocessRecord(
        pid=pid,
        command=command,
        started_at=now,
        last_seen=now,
    )
    with _SUBPROCESS_LOCK:
        _SUBPROCESS_REGISTRY[pid] = record


def unregister_subprocess(*, pid: int, exit_code: int | None = None) -> None:
    """Remove a subprocess from the teardown registry.

    Parameters
    ----------
    pid
        Process identifier for the subprocess to remove.
    exit_code
        Optional exit code for the completed subprocess.
    """
    _ = exit_code
    with _SUBPROCESS_LOCK:
        _SUBPROCESS_REGISTRY.pop(pid, None)


def mark_subprocess_exited(*, pid: int, exit_code: int | None = None) -> None:
    """Mark a subprocess as exited while retaining it for teardown reporting.

    Parameters
    ----------
    pid
        Process identifier for the subprocess to mark.
    exit_code
        Optional exit code for the completed subprocess.
    """
    now = monotonic()
    with _SUBPROCESS_LOCK:
        record = _SUBPROCESS_REGISTRY.get(pid)
        if record is None:
            return
        duration_ms = (now - record.started_at) * 1000
        _SUBPROCESS_REGISTRY[pid] = replace(
            record,
            exit_code=exit_code,
            duration_ms=duration_ms,
            last_seen=now,
        )


def _prune_stale_records(*, now: float) -> None:
    stale_pids = [
        pid
        for pid, record in _SUBPROCESS_REGISTRY.items()
        if record.exit_code is not None and (now - record.last_seen) > _STALE_COMPLETED_TTL_S
    ]
    for pid in stale_pids:
        _SUBPROCESS_REGISTRY.pop(pid, None)


def count_subprocesses() -> int:
    """Return the number of tracked subprocesses.

    Returns
    -------
    int
        Count of subprocess records currently registered.
    """
    now = monotonic()
    with _SUBPROCESS_LOCK:
        _prune_stale_records(now=now)
        return sum(record.exit_code is None for record in _SUBPROCESS_REGISTRY.values())


def snapshot_subprocesses(
    *,
    limit: int | None = None,
    include_completed: bool = False,
) -> tuple[SubprocessRecord, ...]:
    """Return a bounded snapshot of tracked subprocesses.

    Parameters
    ----------
    limit
        Maximum number of subprocess records to return.
    include_completed
        Whether to include completed subprocesses with exit codes.

    Returns
    -------
    tuple[SubprocessRecord, ...]
        Snapshot of the currently registered subprocess records.
    """
    now = monotonic()
    with _SUBPROCESS_LOCK:
        _prune_stale_records(now=now)
        records: list[SubprocessRecord] = []
        for record in _SUBPROCESS_REGISTRY.values():
            if record.exit_code is not None:
                if include_completed:
                    records.append(record)
                continue
            updated = replace(
                record,
                last_seen=now,
                duration_ms=(now - record.started_at) * 1000,
            )
            _SUBPROCESS_REGISTRY[record.pid] = updated
            records.append(updated)
    if limit is not None and limit >= 0:
        records = records[:limit]
    return tuple(records)


__all__ = [
    "SubprocessRecord",
    "count_subprocesses",
    "mark_subprocess_exited",
    "register_subprocess",
    "snapshot_subprocesses",
    "unregister_subprocess",
]
