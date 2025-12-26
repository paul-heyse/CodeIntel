"""Teardown telemetry primitives and emitters."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.observability.attribute_sanitizer import (
    SpanAttributeValue,
    prune_none,
    redact_command_value,
    redact_path_value,
)
from codeintel.observability.attribute_schema import build_attribute_normalizer
from codeintel.observability.events import TelemetryEvent, emit_event
from codeintel.observability.runtime import get_observability
from codeintel.observability.runtime_registry import count_subprocesses, snapshot_subprocesses
from codeintel.observability.semconv_keys import (
    BUILD_COMMIT,
    BUILD_DECISION_TRACE_ARTIFACT,
    BUILD_DURATION_MS,
    BUILD_REPO,
    BUILD_RUN_ID,
    BUILD_SCHEMA_INFERENCE_ERRORS_COUNT,
    BUILD_TARGETS,
    BUILD_VALIDATION_ISSUE_COUNT,
    BUILD_VALIDATION_MODE,
    CLI_COMMAND,
    CLI_ERROR_TYPE,
    CLI_EXIT_CODE,
    CLI_INVOCATION_ID,
    CLI_IS_PARSE_ERROR,
    CODEINTEL_COMPONENT,
    CODEINTEL_DOMAIN,
    CODEINTEL_OPERATION,
    SCIP_COMMIT,
    SCIP_DURATION_MS,
    SCIP_ERROR,
    SCIP_MODE,
    SCIP_REPO,
    SCIP_RUN_ID,
    SCIP_STATUS,
    SHUTDOWN_ACTIVE_THREAD_NAMES,
    SHUTDOWN_ACTIVE_THREADS_COUNT,
    SHUTDOWN_ERROR_MESSAGE,
    SHUTDOWN_ERROR_TYPE,
    SHUTDOWN_PENDING_TASK_SAMPLES,
    SHUTDOWN_PENDING_TASKS_COUNT,
    SHUTDOWN_STATUS,
    SHUTDOWN_SUBPROCESS_COUNT,
    SHUTDOWN_SUBPROCESS_SAMPLES,
    TELEMETRY_FLUSH_MS,
    TELEMETRY_FLUSH_OK,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opentelemetry.trace import Span

log = logging.getLogger(__name__)

ShutdownStatus = Literal["unknown", "failed", "partial", "succeeded"]
ScipTeardownStatus = Literal["unknown", "failed", "skipped", "succeeded"]


@dataclass(frozen=True, slots=True)
class SubprocessSample:
    """Snapshot of a tracked subprocess for teardown telemetry."""

    pid: int
    command: str

    def to_payload(self) -> dict[str, SpanAttributeValue]:
        """Return a JSON-ready payload for the subprocess sample.

        Returns
        -------
        dict[str, SpanAttributeValue]
            JSON-serializable subprocess payload.
        """
        return prune_none({"pid": self.pid, "command": _redact_command_value(self.command)})


@dataclass(frozen=True, slots=True)
class ArtifactSummary:
    """Summarized artifact metadata for teardown telemetry."""

    name: str
    artifact_type: str
    path: str | None
    size_bytes: int | None = None

    def to_payload(self) -> dict[str, SpanAttributeValue]:
        """Return a JSON-ready payload for the artifact summary.

        Returns
        -------
        dict[str, SpanAttributeValue]
            JSON-serializable artifact payload.
        """
        return prune_none(
            {
                "name": self.name,
                "artifact_type": self.artifact_type,
                "path": _redact_path_value(self.path),
                "size_bytes": self.size_bytes,
            }
        )


@dataclass(frozen=True, slots=True)
class TeardownTelemetry:
    """Telemetry payload for teardown logging and spans."""

    component: str = "build"
    operation: str = "shutdown"
    run_id: str | None = None
    repo: str | None = None
    commit: str | None = None
    targets: tuple[str, ...] = field(default_factory=tuple)
    duration_ms: float | None = None
    cli_invocation_id: str | None = None
    cli_command: str | None = None
    cli_exit_code: int | None = None
    cli_is_parse_error: bool | None = None
    cli_error_type: str | None = None
    domain: str | None = None
    decision_trace_artifact: ArtifactSummary | None = None
    decision_trace_path: str | None = None
    validation_mode: str | None = None
    validation_issue_count: int | None = None
    schema_inference_errors_count: int | None = None
    shutdown_status: ShutdownStatus = "unknown"
    pending_tasks_count: int | None = None
    pending_task_samples: tuple[str, ...] = field(default_factory=tuple)
    active_threads_count: int | None = None
    active_thread_names: tuple[str, ...] = field(default_factory=tuple)
    subprocess_count: int | None = None
    subprocess_samples: tuple[SubprocessSample, ...] = field(default_factory=tuple)
    telemetry_flush_ok: bool | None = None
    telemetry_flush_ms: float | None = None

    def span_attributes(self) -> dict[str, SpanAttributeValue]:
        """Return low-cardinality span attributes for teardown telemetry.

        Returns
        -------
        dict[str, SpanAttributeValue]
            Span attribute payload for teardown telemetry.
        """
        targets = ",".join(self.targets) if self.targets else None
        return prune_none(
            {
                CODEINTEL_COMPONENT: self.component,
                CODEINTEL_OPERATION: self.operation,
                BUILD_RUN_ID: self.run_id,
                BUILD_REPO: self.repo,
                BUILD_COMMIT: self.commit,
                BUILD_TARGETS: targets,
                BUILD_DURATION_MS: self.duration_ms,
                CLI_INVOCATION_ID: self.cli_invocation_id,
                CLI_COMMAND: _redact_command_value(self.cli_command),
                CLI_EXIT_CODE: self.cli_exit_code,
                CLI_IS_PARSE_ERROR: self.cli_is_parse_error,
                CLI_ERROR_TYPE: self.cli_error_type,
                CODEINTEL_DOMAIN: self.domain,
                BUILD_DECISION_TRACE_ARTIFACT: self._decision_trace_name(),
                BUILD_VALIDATION_MODE: self.validation_mode,
                BUILD_VALIDATION_ISSUE_COUNT: self.validation_issue_count,
                BUILD_SCHEMA_INFERENCE_ERRORS_COUNT: self.schema_inference_errors_count,
                SHUTDOWN_STATUS: self.shutdown_status,
                SHUTDOWN_PENDING_TASKS_COUNT: self.pending_tasks_count,
                SHUTDOWN_ACTIVE_THREADS_COUNT: self.active_threads_count,
                SHUTDOWN_SUBPROCESS_COUNT: self.subprocess_count,
                TELEMETRY_FLUSH_OK: self.telemetry_flush_ok,
                TELEMETRY_FLUSH_MS: self.telemetry_flush_ms,
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
            f"{sample.pid}:{_redact_command_value(sample.command)}"
            for sample in self.subprocess_samples
        ]
        return prune_none(
            {
                SHUTDOWN_PENDING_TASK_SAMPLES: [*self.pending_task_samples],
                SHUTDOWN_ACTIVE_THREAD_NAMES: [*self.active_thread_names],
                SHUTDOWN_SUBPROCESS_SAMPLES: subprocess_samples or None,
            }
        )

    def to_log_payload(self) -> dict[str, object]:
        """Return a JSON-serializable payload for structured logs.

        Returns
        -------
        dict[str, object]
            Structured log payload for teardown telemetry.
        """
        return {
            "component": self.component,
            "operation": self.operation,
            "run_id": self.run_id,
            "repo": self.repo,
            "commit": self.commit,
            "targets": [*self.targets],
            "duration_ms": self.duration_ms,
            "cli_invocation_id": self.cli_invocation_id,
            "cli_command": _redact_command_value(self.cli_command),
            "cli_exit_code": self.cli_exit_code,
            "cli_is_parse_error": self.cli_is_parse_error,
            "cli_error_type": self.cli_error_type,
            "domain": self.domain,
            "decision_trace_artifact": self._decision_trace_payload(),
            "decision_trace_path": _redact_path_value(self.decision_trace_path),
            "validation_mode": self.validation_mode,
            "validation_issue_count": self.validation_issue_count,
            "schema_inference_errors_count": self.schema_inference_errors_count,
            "shutdown_status": self.shutdown_status,
            "pending_tasks_count": self.pending_tasks_count,
            "pending_task_samples": [*self.pending_task_samples],
            "active_threads_count": self.active_threads_count,
            "active_thread_names": [*self.active_thread_names],
            "subprocess_count": self.subprocess_count,
            "subprocess_samples": [sample.to_payload() for sample in self.subprocess_samples],
            "telemetry_flush_ok": self.telemetry_flush_ok,
            "telemetry_flush_ms": self.telemetry_flush_ms,
        }

    def shutdown_error_attributes(self) -> dict[str, SpanAttributeValue]:
        """Return span event attributes for shutdown errors.

        Returns
        -------
        dict[str, SpanAttributeValue]
            Span attributes for shutdown errors.
        """
        return prune_none(
            {
                SHUTDOWN_STATUS: self.shutdown_status,
                CLI_ERROR_TYPE: self.cli_error_type,
                CLI_EXIT_CODE: self.cli_exit_code,
            }
        )

    def _decision_trace_name(self) -> str | None:
        if self.decision_trace_artifact is None:
            return None
        return self.decision_trace_artifact.name

    def _decision_trace_payload(self) -> Mapping[str, SpanAttributeValue] | None:
        if self.decision_trace_artifact is None:
            return None
        return self.decision_trace_artifact.to_payload()


@dataclass(frozen=True, slots=True)
class TeardownSnapshot:
    """Snapshot of runtime state for teardown telemetry."""

    pending_tasks_count: int | None
    pending_task_samples: tuple[str, ...]
    active_threads_count: int | None
    active_thread_names: tuple[str, ...]
    subprocess_count: int | None
    subprocess_samples: tuple[SubprocessSample, ...]
    telemetry_flush_ok: bool | None
    telemetry_flush_ms: float | None


@dataclass(frozen=True, slots=True)
class TeardownSnapshotOptions:
    """Options controlling teardown snapshot collection."""

    task_sample_limit: int | None
    thread_sample_limit: int | None
    subprocess_sample_limit: int | None
    allowlisted_daemon_names: set[str] | None
    telemetry_flush_ok: bool | None = None
    telemetry_flush_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ScipTeardownTelemetry:
    """Telemetry payload for SCIP teardown logging and spans."""

    run_id: str | None = None
    repo: str | None = None
    commit: str | None = None
    scip_mode: str | None = None
    status: ScipTeardownStatus = "unknown"
    error_summary: str | None = None
    duration_ms: float | None = None

    def shutdown_error_attributes(self) -> dict[str, SpanAttributeValue]:
        """Return span event attributes for SCIP teardown errors.

        Returns
        -------
        dict[str, SpanAttributeValue]
            Span attributes for SCIP teardown errors.
        """
        return prune_none(
            {
                SCIP_STATUS: self.status,
                SCIP_ERROR: self.error_summary,
            }
        )

    def span_attributes(self) -> dict[str, SpanAttributeValue]:
        """Return low-cardinality span attributes for SCIP teardown telemetry.

        Returns
        -------
        dict[str, SpanAttributeValue]
            Span attributes with None values pruned.
        """
        return prune_none(
            {
                SCIP_RUN_ID: self.run_id,
                SCIP_REPO: self.repo,
                SCIP_COMMIT: self.commit,
                SCIP_MODE: self.scip_mode,
                SCIP_STATUS: self.status,
                SCIP_ERROR: self.error_summary,
                SCIP_DURATION_MS: self.duration_ms,
            }
        )

    def to_log_payload(self) -> dict[str, object]:
        """Return a JSON-serializable payload for structured logs.

        Returns
        -------
        dict[str, object]
            Payload suitable for structured logging.
        """
        return {
            "run_id": self.run_id,
            "repo": self.repo,
            "commit": self.commit,
            "scip_mode": self.scip_mode,
            "status": self.status,
            "error_summary": self.error_summary,
            "duration_ms": self.duration_ms,
        }


def snapshot_pending_tasks(
    *,
    sample_limit: int | None = None,
) -> tuple[int | None, tuple[str, ...]]:
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


def collect_teardown_snapshot(options: TeardownSnapshotOptions) -> TeardownSnapshot:
    """Collect teardown snapshot state for telemetry.

    Returns
    -------
    TeardownSnapshot
        Snapshot of pending tasks, threads, subprocesses, and flush metadata.
    """
    pending_tasks_count, pending_task_samples = snapshot_pending_tasks(
        sample_limit=options.task_sample_limit,
    )
    active_threads_count, active_thread_names = snapshot_active_threads(
        sample_limit=options.thread_sample_limit,
        allowlisted_daemon_names=options.allowlisted_daemon_names,
    )
    subprocess_records = snapshot_subprocesses(limit=options.subprocess_sample_limit)
    subprocess_samples = tuple(
        SubprocessSample(pid=record.pid, command=record.command) for record in subprocess_records
    )
    return TeardownSnapshot(
        pending_tasks_count=pending_tasks_count,
        pending_task_samples=pending_task_samples,
        active_threads_count=active_threads_count,
        active_thread_names=active_thread_names,
        subprocess_count=count_subprocesses(),
        subprocess_samples=subprocess_samples,
        telemetry_flush_ok=options.telemetry_flush_ok,
        telemetry_flush_ms=options.telemetry_flush_ms,
    )


def emit_teardown_telemetry(
    telemetry: TeardownTelemetry,
    *,
    span_name: str = "build.shutdown",
    logger: logging.Logger | None = None,
) -> None:
    """Emit teardown telemetry via OpenTelemetry spans and structured logs.

    Parameters
    ----------
    telemetry
        Teardown telemetry payload to emit.
    span_name
        Span name to use for teardown span.
    logger
        Optional logger override.
    """
    obs = get_observability()
    span: Span | None = None
    normalizer = build_attribute_normalizer(obs.policy)
    log_level = logging.INFO if telemetry.shutdown_status == "succeeded" else logging.WARNING
    if obs.enabled and obs.tracer is not None:
        with obs.tracer.start_as_current_span(span_name) as active_span:
            span = active_span
            summary_event = TelemetryEvent(
                name="shutdown.summary",
                span_attributes=telemetry.span_attributes(),
                event_attributes=telemetry.event_attributes(),
                log_payload=telemetry.to_log_payload(),
                span_event_name="shutdown.summary",
                log_event_name="build.shutdown",
                log_level=log_level,
            )
            emit_event(
                event=summary_event,
                span=span,
                normalizer=normalizer,
                logger=logger,
            )
            if telemetry.shutdown_status != "succeeded":
                error_event = TelemetryEvent(
                    name="shutdown.error",
                    span_attributes=telemetry.span_attributes(),
                    event_attributes=telemetry.shutdown_error_attributes(),
                    log_payload=telemetry.to_log_payload(),
                    span_event_name="shutdown.error",
                    log_event_name="build.shutdown",
                    log_level=log_level,
                )
                emit_event(
                    event=error_event,
                    span=span,
                    normalizer=normalizer,
                    logger=logger,
                )
            return

    summary_event = TelemetryEvent(
        name="shutdown.summary",
        span_attributes=telemetry.span_attributes(),
        event_attributes=telemetry.event_attributes(),
        log_payload=telemetry.to_log_payload(),
        span_event_name="shutdown.summary",
        log_event_name="build.shutdown",
        log_level=log_level,
    )
    emit_event(
        event=summary_event,
        span=None,
        normalizer=normalizer,
        logger=logger,
    )


def emit_scip_teardown_telemetry(
    telemetry: ScipTeardownTelemetry,
    *,
    span_name: str = "scip.teardown",
    logger: logging.Logger | None = None,
) -> None:
    """Emit SCIP teardown telemetry via OpenTelemetry spans and structured logs.

    Parameters
    ----------
    telemetry
        SCIP teardown telemetry payload to emit.
    span_name
        Span name to use for teardown span.
    logger
        Optional logger override.
    """
    obs = get_observability()
    span: Span | None = None
    normalizer = build_attribute_normalizer(obs.policy)
    log_level = logging.INFO if telemetry.status == "succeeded" else logging.WARNING
    if obs.enabled and obs.tracer is not None:
        with obs.tracer.start_as_current_span(span_name) as active_span:
            span = active_span
            summary_event = TelemetryEvent(
                name="scip.teardown",
                span_attributes=telemetry.span_attributes(),
                event_attributes={},
                log_payload=telemetry.to_log_payload(),
                span_event_name="scip.teardown",
                log_event_name="scip.teardown",
                log_level=log_level,
            )
            emit_event(
                event=summary_event,
                span=span,
                normalizer=normalizer,
                logger=logger,
            )
            if telemetry.status != "succeeded":
                error_event = TelemetryEvent(
                    name="shutdown.error",
                    span_attributes=telemetry.span_attributes(),
                    event_attributes=telemetry.shutdown_error_attributes(),
                    log_payload=telemetry.to_log_payload(),
                    span_event_name="shutdown.error",
                    log_event_name="scip.teardown",
                    log_level=log_level,
                )
                emit_event(
                    event=error_event,
                    span=span,
                    normalizer=normalizer,
                    logger=logger,
                )
            return

    summary_event = TelemetryEvent(
        name="scip.teardown",
        span_attributes=telemetry.span_attributes(),
        event_attributes={},
        log_payload=telemetry.to_log_payload(),
        span_event_name="scip.teardown",
        log_event_name="scip.teardown",
        log_level=log_level,
    )
    emit_event(
        event=summary_event,
        span=None,
        normalizer=normalizer,
        logger=logger,
    )


def emit_shutdown_error_event(
    *,
    span_name: str,
    error: Exception,
    logger: logging.Logger | None = None,
    attributes: Mapping[str, SpanAttributeValue] | None = None,
) -> None:
    """Emit a shutdown.error span event and structured log entry.

    Parameters
    ----------
    span_name
        Span name to use for the shutdown error span.
    error
        Exception to record.
    logger
        Optional logger override.
    attributes
        Additional span attributes to attach.
    """
    obs = get_observability()
    event_attrs = dict(attributes or {})
    event_attrs[SHUTDOWN_ERROR_TYPE] = type(error).__name__
    event_attrs[SHUTDOWN_ERROR_MESSAGE] = str(error)
    normalizer = build_attribute_normalizer(obs.policy)
    event = TelemetryEvent(
        name="shutdown.error",
        span_attributes=event_attrs,
        event_attributes=event_attrs,
        log_payload=prune_none(event_attrs),
        span_event_name="shutdown.error",
        log_event_name="shutdown.error",
        log_level=logging.WARNING,
    )
    if obs.enabled and obs.tracer is not None:
        with obs.tracer.start_as_current_span(span_name) as span:
            emit_event(
                event=event,
                span=span,
                normalizer=normalizer,
                logger=logger,
            )
    else:
        emit_event(
            event=event,
            span=None,
            normalizer=normalizer,
            logger=logger,
        )


def _redact_command_value(value: str | None) -> str | None:
    policy = get_observability().policy
    return redact_command_value(value, keep_segments=policy.redaction.command_keep_segments)


def _redact_path_value(value: str | None) -> str | None:
    policy = get_observability().policy
    return redact_path_value(value, keep_segments=policy.redaction.path_keep_segments)


__all__ = [
    "ScipTeardownTelemetry",
    "SubprocessSample",
    "TeardownSnapshot",
    "TeardownSnapshotOptions",
    "TeardownTelemetry",
    "collect_teardown_snapshot",
    "emit_scip_teardown_telemetry",
    "emit_shutdown_error_event",
    "emit_teardown_telemetry",
    "snapshot_active_threads",
    "snapshot_pending_tasks",
]
