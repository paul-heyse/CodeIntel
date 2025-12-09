"""Protocol-aligned ingestion/build fakes for realistic test wiring.

These fakes implement the public interfaces used in production so tests can
exercise real DuckDB-backed storage and build orchestration without resorting
to monkeypatching. They record calls for assertions while delegating to the
real implementations when provided.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from codeintel.build.executor import BuildExecutor, BuildResult
from codeintel.build.plan import BuildPlan
from codeintel.ingestion.adapters.duckdb_storage import IngestStorageService
from codeintel.storage.gateway import StorageGateway

BatchCall = tuple[str, Sequence[Sequence[object]], Sequence[object] | None, str | None]
"""Recorded call tuple for run_batch invocations."""


@dataclass
class RecordingIngestStorageService:
    """IngestStorageService wrapper that records batches while delegating.

    Parameters
    ----------
    service
        Underlying ingestion storage service to delegate to.
    record_calls
        Optional callback invoked with each call for additional recording.
    """

    service: IngestStorageService
    record_calls: Callable[[BatchCall], None] | None = None
    calls: list[BatchCall] = field(default_factory=list)

    def run_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        delete_params: Sequence[object] | None = None,
        scope: str | None = None,
    ) -> object:
        """Record and delegate batch execution.

        Returns
        -------
        object
            BatchResult from the underlying service.
        """
        call: BatchCall = (table_key, rows, delete_params, scope)
        self.calls.append(call)
        if self.record_calls is not None:
            self.record_calls(call)
        return self.service.run_batch(
            table_key,
            rows,
            delete_params=delete_params,
            scope=scope,
        )

    @classmethod
    def from_gateway(cls, gateway: StorageGateway) -> RecordingIngestStorageService:
        """Build a recording service from a storage gateway.

        Returns
        -------
        RecordingIngestStorageService
            Service that records calls and writes to the provided gateway.
        """
        return cls(service=IngestStorageService.from_gateway(gateway))


@dataclass
class RecordingBuildExecutor:
    """BuildExecutor stand-in that records execute calls and optionally delegates.

    Parameters
    ----------
    delegate
        Optional real BuildExecutor to delegate execution to for realism.
    run_id_factory
        Optional callable to generate deterministic run_ids for tests.
    """

    delegate: BuildExecutor | None = None
    run_id_factory: Callable[[], str] | None = None
    executions: list[tuple[BuildPlan, bool]] = field(default_factory=list)

    def execute(self, plan: BuildPlan, *, dry_run: bool = False) -> BuildResult:
        """Record execution and delegate or return a minimal success result.

        Returns
        -------
        BuildResult
            Execution result from the delegate or a minimal success result.
        """
        self.executions.append((plan, dry_run))
        if self.delegate is not None:
            return self.delegate.execute(plan, dry_run=dry_run)

        run_id = self.run_id_factory() if self.run_id_factory is not None else "recording"
        duration_ms = 0.0
        return BuildResult(
            run_id=run_id,
            plan=plan,
            status="succeeded",
            completed_targets=tuple(step.target for stage in plan.stages for step in stage.steps),
            failed_targets=(),
            skipped_targets=(),
            duration_ms=duration_ms,
            error_summary=None,
        )


__all__ = [
    "BatchCall",
    "RecordingBuildExecutor",
    "RecordingIngestStorageService",
]
