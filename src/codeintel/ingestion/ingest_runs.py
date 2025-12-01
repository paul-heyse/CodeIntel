"""Control-plane record for ingestion steps with pluggable sinks."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

from codeintel.config.dataset_contract import ingest_run_to_tuple
from codeintel.ingestion.common import run_batch
from codeintel.ingestion.tool_runner import ToolExecutionError, ToolNotFoundError
from codeintel.storage.gateway import DuckDBError, StorageGateway

log = logging.getLogger(__name__)


class _MetricRecorder(Protocol):
    """Minimal interface for OpenTelemetry histogram recorders."""

    def record(self, value: float, attributes: Mapping[str, str]) -> None:
        """Record a single measurement."""
        ...


class IngestRunStatus(StrEnum):
    """Outcome for an ingestion step run."""

    OK = "ok"
    SKIPPED = "skipped"
    ERROR = "error"


class IngestRunMode(StrEnum):
    """High-level mode for a dataset step."""

    FULL = "full"
    INCREMENTAL = "incremental"
    UNKNOWN = "unknown"


@dataclass
class IngestRun:
    """
    Structured record describing a single ingestion step execution.

    Fields are deliberately redundant so they can be shipped directly into JSONL
    or a logging DuckDB without further transformation.
    """

    run_id: str
    repo: str
    commit: str
    step: str
    datasets: tuple[str, ...]
    mode: IngestRunMode
    started_at: datetime
    finished_at: datetime | None = None
    duration_s: float | None = None

    rows_before: Mapping[str, int] = field(default_factory=dict)
    rows_after: Mapping[str, int] = field(default_factory=dict)
    rows_inserted: int = 0
    rows_deleted: int = 0

    status: IngestRunStatus = IngestRunStatus.OK
    error_kind: str | None = None
    error_message: str | None = None

    # Incremental view metrics populated when run_incremental_ingest observers are used.
    modules_total: int | None = None
    modules_changed: int | None = None
    modules_deleted: int | None = None
    modules_changed_ratio: float | None = None
    modules_deleted_ratio: float | None = None
    use_full_rebuild: bool | None = None


class IngestRunSink(Protocol):
    """Abstraction for recording IngestRun objects somewhere."""

    def record(self, run: IngestRun) -> None:
        """Persist or emit the run record."""
        ...


@dataclass
class MultiSink(IngestRunSink):
    """Fan out a run record to multiple sinks."""

    sinks: Sequence[IngestRunSink]

    def record(self, run: IngestRun) -> None:
        """Forward the run record to all configured sinks, ignoring sink failures."""
        for sink in self.sinks:
            try:
                sink.record(run)
            except (
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
            ):  # pragma: no cover - sink errors should not break callers
                log.exception("IngestRun sink failed: sink=%s run_id=%s", sink, run.run_id)


def classify_error(exc: BaseException) -> str:
    """
    Map exceptions into coarse error kinds suitable for dashboards.

    This can be extended over time (e.g. tagging parse errors, validation errors, etc.).

    Parameters
    ----------
    exc
        Exception raised by an ingestion step.

    Returns
    -------
    str
        Normalized error kind string.
    """
    if isinstance(exc, ToolNotFoundError):
        return "tool_not_found"
    if isinstance(exc, ToolExecutionError):
        message = str(exc).lower()
        if "timeout" in message or "timed out" in message:
            return "tool_timeout"
        return "tool_execution"
    if isinstance(exc, DuckDBError):
        return "db_error"
    if isinstance(exc, ValueError):
        return "parse_error"
    return exc.__class__.__name__


@dataclass
class JsonlIngestRunSink:
    """
    Sink that appends IngestRun records as JSON lines on disk.

    Default path suggestion:
        BuildPaths.build_dir / "logs" / "ingest_runs.jsonl"
    """

    path: Path

    def record(self, run: IngestRun) -> None:
        """Append a JSONL line for the provided run."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(run)
        payload["started_at"] = run.started_at.isoformat()
        if run.finished_at is not None:
            payload["finished_at"] = run.finished_at.isoformat()
        with self.path.open("a", encoding="utf8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")


@dataclass
class DuckDBIngestRunSink:
    """Persist IngestRun rows into the configured DuckDB database."""

    gateway: StorageGateway

    def record(self, run: IngestRun) -> None:
        """Insert the run record into core.ingest_runs."""
        row = ingest_run_to_tuple(cast("Any", run))
        run_batch(
            self.gateway,
            "core.ingest_runs",
            [row],
            delete_params=None,
            scope=f"{run.repo}@{run.commit}",
        )


@dataclass
class OtelIngestRunSink(IngestRunSink):
    """Example sink that emits IngestRun metrics via OpenTelemetry."""

    meter_name: str = __name__
    _duration: _MetricRecorder = field(init=False, repr=False)
    _rows_inserted: _MetricRecorder = field(init=False, repr=False)
    _rows_deleted: _MetricRecorder = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Initialize OpenTelemetry recorders if the dependency is installed.

        Raises
        ------
        RuntimeError
            If the optional opentelemetry dependency is missing.
        """
        try:
            metrics_module = importlib.import_module("opentelemetry.metrics")
        except ImportError as exc:  # pragma: no cover - optional dependency
            message = "opentelemetry not installed; OtelIngestRunSink cannot emit metrics"
            raise RuntimeError(message) from exc
        meter = metrics_module.get_meter(self.meter_name)
        self._duration = meter.create_histogram(
            "codeintel.ingest.duration",
            unit="s",
            description="Ingestion step duration in seconds",
        )
        self._rows_inserted = meter.create_histogram(
            "codeintel.ingest.rows_inserted",
            unit="rows",
            description="Rows inserted by an ingestion step",
        )
        self._rows_deleted = meter.create_histogram(
            "codeintel.ingest.rows_deleted",
            unit="rows",
            description="Rows deleted by an ingestion step",
        )

    def record(self, run: IngestRun) -> None:
        """Emit metrics for the provided run."""
        labels = {
            "repo": run.repo,
            "step": run.step,
            "status": run.status.value,
            "mode": run.mode.value,
        }
        if run.duration_s is not None:
            self._duration.record(run.duration_s, labels)
        self._rows_inserted.record(run.rows_inserted, labels)
        self._rows_deleted.record(run.rows_deleted, labels)
