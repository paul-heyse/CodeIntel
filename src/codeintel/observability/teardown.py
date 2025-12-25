"""Teardown telemetry primitives and emitters."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.observability.otel import get_observability

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opentelemetry.trace import Span

log = logging.getLogger(__name__)

SpanAttributeValue = (
    str
    | bool
    | int
    | float
    | list[str]
    | list[bool]
    | list[int]
    | list[float]
)


@dataclass(frozen=True, slots=True)
class SubprocessSample:
    """Snapshot of a tracked subprocess for teardown telemetry."""

    pid: int
    command: str

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-ready payload for the subprocess sample.

        Returns
        -------
        dict[str, object]
            JSON-serializable subprocess payload.
        """
        return {"pid": self.pid, "command": self.command}


@dataclass(frozen=True, slots=True)
class TeardownTelemetry:
    """Telemetry payload for teardown logging and spans."""

    run_id: str | None
    repo: str | None
    commit: str | None
    targets: tuple[str, ...] = field(default_factory=tuple)
    duration_ms: float | None = None
    shutdown_status: str = "unknown"
    pending_tasks_count: int | None = None
    pending_task_samples: tuple[str, ...] = field(default_factory=tuple)
    active_threads_count: int | None = None
    active_thread_names: tuple[str, ...] = field(default_factory=tuple)
    subprocess_count: int | None = None
    subprocess_samples: tuple[SubprocessSample, ...] = field(default_factory=tuple)
    telemetry_flush_status: str | None = None
    telemetry_flush_ms: float | None = None

    def span_attributes(self) -> dict[str, SpanAttributeValue]:
        """Return low-cardinality span attributes for teardown telemetry.

        Returns
        -------
        dict[str, SpanAttributeValue]
            Span attribute payload for teardown telemetry.
        """
        targets = ",".join(self.targets) if self.targets else None
        return _prune_none(
            {
                "build.run_id": self.run_id,
                "build.repo": self.repo,
                "build.commit": self.commit,
                "build.targets": targets,
                "build.duration_ms": self.duration_ms,
                "shutdown.status": self.shutdown_status,
                "shutdown.pending_tasks_count": self.pending_tasks_count,
                "shutdown.active_threads_count": self.active_threads_count,
                "shutdown.subprocess_count": self.subprocess_count,
                "telemetry.flush.status": self.telemetry_flush_status,
                "telemetry.flush.ms": self.telemetry_flush_ms,
            }
        )

    def event_attributes(self) -> dict[str, SpanAttributeValue]:
        """Return event attributes with bounded samples for teardown telemetry.

        Returns
        -------
        dict[str, SpanAttributeValue]
            Event attribute payload for teardown telemetry.
        """
        subprocess_samples = [
            f"{sample.pid}:{sample.command}" for sample in self.subprocess_samples
        ]
        return _prune_none(
            {
                "shutdown.pending_task_samples": [*self.pending_task_samples],
                "shutdown.active_thread_names": [*self.active_thread_names],
                "shutdown.subprocess_samples": subprocess_samples or None,
            }
        )

    def to_log_payload(self, *, event: str = "build.shutdown") -> dict[str, object]:
        """Return a JSON-serializable payload for structured logs.

        Returns
        -------
        dict[str, object]
            Structured log payload for teardown telemetry.
        """
        return {
            "event": event,
            "run_id": self.run_id,
            "repo": self.repo,
            "commit": self.commit,
            "targets": [*self.targets],
            "duration_ms": self.duration_ms,
            "shutdown_status": self.shutdown_status,
            "pending_tasks_count": self.pending_tasks_count,
            "pending_task_samples": [*self.pending_task_samples],
            "active_threads_count": self.active_threads_count,
            "active_thread_names": [*self.active_thread_names],
            "subprocess_count": self.subprocess_count,
            "subprocess_samples": [sample.to_payload() for sample in self.subprocess_samples],
            "telemetry_flush_status": self.telemetry_flush_status,
            "telemetry_flush_ms": self.telemetry_flush_ms,
        }


def snapshot_pending_tasks(*, sample_limit: int | None = None) -> tuple[int | None, tuple[str, ...]]:
    """Collect pending asyncio tasks for teardown telemetry.

    Parameters
    ----------
    sample_limit
        Maximum number of task names to return.

    Returns
    -------
    tuple[int | None, tuple[str, ...]]
        Tuple of (pending task count, sample of task names). Count is None when
        no running event loop is available.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None, ()

    tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
    count = len(tasks)
    if not tasks:
        return count, ()

    names = [task.get_name() for task in tasks]
    if sample_limit is not None and sample_limit >= 0:
        names = names[:sample_limit]
    return count, tuple(names)


def snapshot_active_threads(
    *,
    sample_limit: int | None = None,
    allowlisted_daemon_names: set[str] | None = None,
) -> tuple[int, tuple[str, ...]]:
    """Collect active threads for teardown telemetry.

    Parameters
    ----------
    sample_limit
        Maximum number of thread names to return.
    allowlisted_daemon_names
        Daemon thread names that should be excluded from reporting.

    Returns
    -------
    tuple[int, tuple[str, ...]]
        Tuple of (active thread count, sample of thread names).
    """
    allowlist = allowlisted_daemon_names or set()
    threads = [
        thread
        for thread in threading.enumerate()
        if thread.is_alive() and not (thread.daemon and thread.name in allowlist)
    ]
    count = len(threads)
    names = [thread.name for thread in threads]
    if sample_limit is not None and sample_limit >= 0:
        names = names[:sample_limit]
    return count, tuple(names)


def emit_teardown_telemetry(
    telemetry: TeardownTelemetry,
    *,
    span_name: str = "build.shutdown",
    logger: logging.Logger | None = None,
) -> None:
    """Emit teardown telemetry via OpenTelemetry spans and structured logs."""
    obs = get_observability()
    span: Span | None = None
    if obs.enabled and obs.tracer is not None:
        with obs.tracer.start_as_current_span(span_name) as active_span:
            span = active_span
            _set_span_attributes(span, telemetry.span_attributes())
            _add_span_event(span, "shutdown.summary", telemetry.event_attributes())
    log_payload = telemetry.to_log_payload()
    log_target = logger or log
    log_target.info("build.shutdown %s", json.dumps(log_payload, sort_keys=True))

def _set_span_attributes(span: Span, attributes: Mapping[str, SpanAttributeValue]) -> None:
    for key, value in attributes.items():
        attr_value = _coerce_span_value(value)
        if attr_value is not None:
            span.set_attribute(key, attr_value)


def _add_span_event(
    span: Span,
    name: str,
    attributes: Mapping[str, SpanAttributeValue],
) -> None:
    event_attrs: dict[str, SpanAttributeValue] = {}
    for key, value in attributes.items():
        attr_value = _coerce_span_value(value)
        if attr_value is not None:
            event_attrs[key] = attr_value
    if event_attrs:
        span.add_event(name, attributes=event_attrs)


def _coerce_span_value(value: object) -> SpanAttributeValue | None:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (str, bool, int, float)) for item in value):
            return list(value)
        return [str(item) for item in value]
    return str(value)


def _prune_none(values: Mapping[str, SpanAttributeValue | None]) -> dict[str, SpanAttributeValue]:
    return {key: value for key, value in values.items() if value is not None}


__all__ = [
    "SubprocessSample",
    "TeardownTelemetry",
    "emit_teardown_telemetry",
    "snapshot_active_threads",
    "snapshot_pending_tasks",
]
