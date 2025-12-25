"""Runtime registries for teardown telemetry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from time import monotonic

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SubprocessRecord:
    """Tracked subprocess metadata for teardown reporting."""

    pid: int
    command: str
    started_at: float


_SUBPROCESS_LOCK = Lock()
_SUBPROCESS_REGISTRY: dict[int, SubprocessRecord] = {}


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
    record = SubprocessRecord(pid=pid, command=command, started_at=monotonic())
    with _SUBPROCESS_LOCK:
        _SUBPROCESS_REGISTRY[pid] = record


def unregister_subprocess(*, pid: int) -> None:
    """Remove a subprocess from the teardown registry.

    Parameters
    ----------
    pid
        Process identifier for the subprocess to remove.
    """
    with _SUBPROCESS_LOCK:
        _SUBPROCESS_REGISTRY.pop(pid, None)


def count_subprocesses() -> int:
    """Return the number of tracked subprocesses.

    Returns
    -------
    int
        Count of subprocess records currently registered.
    """
    with _SUBPROCESS_LOCK:
        return len(_SUBPROCESS_REGISTRY)


def snapshot_subprocesses(*, limit: int | None = None) -> tuple[SubprocessRecord, ...]:
    """Return a bounded snapshot of tracked subprocesses.

    Parameters
    ----------
    limit
        Maximum number of subprocess records to return.

    Returns
    -------
    tuple[SubprocessRecord, ...]
        Snapshot of the currently registered subprocess records.
    """
    with _SUBPROCESS_LOCK:
        records = list(_SUBPROCESS_REGISTRY.values())
    if limit is not None and limit >= 0:
        records = records[:limit]
    return tuple(records)


__all__ = [
    "SubprocessRecord",
    "count_subprocesses",
    "register_subprocess",
    "snapshot_subprocesses",
    "unregister_subprocess",
]
