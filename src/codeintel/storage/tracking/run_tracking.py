"""Pipeline run tracking persistence for DuckDB.

This module provides persistent tracking of pipeline runs and their steps,
enabling observability and debugging of ingestion, graphs, analytics, and export
executions.

All DuckDB access is encapsulated here, following the storage layer pattern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from codeintel.storage.helpers.json import (
    deserialize_str_tuple,
    serialize_str_sequence,
)
from codeintel.storage.helpers.time import utc_now

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from duckdb import DuckDBPyConnection

    from codeintel.core.execution import RunContext

PipelineStatus = Literal["running", "succeeded", "failed", "partial"]
"""Status of a pipeline run.

- ``running``: Run is currently in progress
- ``succeeded``: Run completed successfully
- ``failed``: Run failed with errors
- ``partial``: Run completed with some steps failed or skipped
"""

StepStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]
"""Status of a pipeline step.

- ``pending``: Step has not started
- ``running``: Step is currently executing
- ``succeeded``: Step completed successfully
- ``failed``: Step failed with an error
- ``skipped``: Step was skipped (e.g., unchanged inputs)
"""

ModuleKind = Literal["ingestion", "graphs", "analytics", "export"]
"""Classification of pipeline module."""


@dataclass(frozen=True)
class PipelineRunRecord:
    """Record of a pipeline run.

    Parameters
    ----------
    run_id
        Unique identifier for this run.
    repo
        Repository slug.
    commit
        Commit SHA.
    kind
        Run kind (ingest, graphs, analytics, full, op_prereqs).
    trigger
        How the run was triggered.
    status
        Current status of the run.
    started_at
        When the run started.
    completed_at
        When the run completed (None if still running).
    requested_operation
        Optional operation that triggered this run.
    requested_datasets
        Datasets requested for this run.
    error_summary
        Summary of errors if failed.
    pipeline_name
        Optional user-facing pipeline name.
    """

    run_id: str
    repo: str
    commit: str
    kind: str
    trigger: str
    status: PipelineStatus
    started_at: datetime
    completed_at: datetime | None = None
    requested_operation: str | None = None
    requested_datasets: tuple[str, ...] = ()
    error_summary: str | None = None
    pipeline_name: str | None = None


@dataclass(frozen=True)
class PipelineStepRecord:
    """Record of a pipeline step execution.

    Parameters
    ----------
    run_id
        Parent run identifier.
    module
        Module that executed this step (ingestion, graphs, analytics, export).
    stage
        Stage within the module (e.g., scan, parse, index).
    name
        Step name (typically the plugin name).
    status
        Current status of the step.
    started_at
        When the step started.
    completed_at
        When the step completed (None if still running).
    row_counts
        Mapping of table names to row counts written.
    extra
        Additional metadata from execution.
    """

    run_id: str
    module: ModuleKind
    stage: str
    name: str
    status: StepStatus
    started_at: datetime
    completed_at: datetime | None = None
    row_counts: Mapping[str, int] | None = None
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class StepCompletionParams:
    """Parameters for completing a pipeline step.

    Bundles completion data to reduce function argument count.

    Parameters
    ----------
    run_id
        Parent run identifier.
    module
        Module that executed the step.
    stage
        Stage within the module.
    name
        Step name (typically plugin name).
    status
        Final status of the step.
    started_at
        When the step started (from start_step).
    row_counts
        Optional mapping of table names to row counts.
    extra
        Optional additional metadata.
    """

    run_id: str
    module: ModuleKind
    stage: str
    name: str
    status: StepStatus
    started_at: datetime
    row_counts: Mapping[str, int] | None = None
    extra: Mapping[str, Any] | None = None

    def to_record(self) -> PipelineStepRecord:
        """Convert to a PipelineStepRecord with current timestamp.

        Returns
        -------
        PipelineStepRecord
            Step record with completed_at set to now.
        """
        return PipelineStepRecord(
            run_id=self.run_id,
            module=self.module,
            stage=self.stage,
            name=self.name,
            status=self.status,
            started_at=self.started_at,
            completed_at=utc_now(),
            row_counts=self.row_counts,
            extra=self.extra,
        )


@dataclass(frozen=True)
class PipelineRunTracking:
    """Pipeline run tracking accessors for the storage gateway.

    This class encapsulates all DuckDB operations for pipeline run tracking,
    following the same pattern as CoreTables, GraphTables, etc.
    """

    con: DuckDBPyConnection

    def start_run(
        self,
        ctx: RunContext,
        *,
        pipeline_name: str | None = None,
        status: PipelineStatus = "running",
    ) -> None:
        """Create or update a pipeline run record.

        Insert a new run or replace an existing one with the same run_id.
        Typically called at the beginning of an orchestrated run.

        Parameters
        ----------
        ctx
            RunContext containing run metadata.
        pipeline_name
            Optional user-facing name for the pipeline.
        status
            Initial status (default: "running").
        """
        datasets_json = serialize_str_sequence(ctx.requested_datasets)
        self.con.execute(
            """
            INSERT OR REPLACE INTO metadata.pipeline_runs (
                run_id,
                repo,
                commit,
                kind,
                trigger,
                requested_operation,
                requested_datasets,
                started_at,
                completed_at,
                status,
                error_summary,
                pipeline_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ctx.run_id,
                ctx.snapshot.repo,
                ctx.snapshot.commit,
                ctx.kind,
                ctx.trigger,
                ctx.requested_operation,
                datasets_json,
                utc_now(),
                None,
                status,
                None,
                pipeline_name,
            ],
        )

    def complete_run(
        self,
        run_id: str,
        *,
        status: PipelineStatus,
        error_summary: str | None = None,
    ) -> None:
        """Mark a pipeline run as completed.

        Update the run record with final status and completion timestamp.

        Parameters
        ----------
        run_id
            Run identifier to update.
        status
            Final status of the run.
        error_summary
            Optional summary of errors if failed.
        """
        self.con.execute(
            """
            UPDATE metadata.pipeline_runs
            SET status = ?,
                error_summary = ?,
                completed_at = ?
            WHERE run_id = ?
            """,
            [status, error_summary, utc_now(), run_id],
        )

    def fetch_run(self, run_id: str) -> PipelineRunRecord | None:
        """Fetch a pipeline run record by ID.

        Parameters
        ----------
        run_id
            Run identifier to fetch.

        Returns
        -------
        PipelineRunRecord | None
            The run record if found, None otherwise.
        """
        cur = self.con.execute(
            """
            SELECT
                run_id,
                repo,
                commit,
                kind,
                trigger,
                requested_operation,
                requested_datasets,
                started_at,
                completed_at,
                status,
                error_summary,
                pipeline_name
            FROM metadata.pipeline_runs
            WHERE run_id = ?
            """,
            [run_id],
        )
        row = cur.fetchone()
        if row is None:
            return None

        (
            run_id_val,
            repo,
            commit,
            kind,
            trigger,
            requested_operation,
            requested_datasets_raw,
            started_at,
            completed_at,
            status,
            error_summary,
            pipeline_name,
        ) = row

        return PipelineRunRecord(
            run_id=str(run_id_val),
            repo=str(repo),
            commit=str(commit),
            kind=str(kind),
            trigger=str(trigger),
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            requested_operation=str(requested_operation) if requested_operation else None,
            requested_datasets=deserialize_str_tuple(requested_datasets_raw),
            error_summary=str(error_summary) if error_summary else None,
            pipeline_name=str(pipeline_name) if pipeline_name else None,
        )

    def record_step(self, record: PipelineStepRecord) -> None:
        """Insert or replace a pipeline step record.

        Can be called once at the end of a step, or twice (at start with
        status='running' and at end with final status).

        Parameters
        ----------
        record
            Step record to persist.
        """
        row_counts_json = (
            json.dumps(dict(record.row_counts), separators=(",", ":"))
            if record.row_counts
            else None
        )
        extra_json = json.dumps(dict(record.extra), separators=(",", ":")) if record.extra else None

        self.con.execute(
            """
            INSERT OR REPLACE INTO metadata.pipeline_steps (
                run_id,
                module,
                stage,
                name,
                started_at,
                completed_at,
                status,
                row_counts,
                extra
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.run_id,
                record.module,
                record.stage,
                record.name,
                record.started_at,
                record.completed_at,
                record.status,
                row_counts_json,
                extra_json,
            ],
        )

    def fetch_steps(self, run_id: str) -> list[PipelineStepRecord]:
        """Fetch all step records for a run.

        Parameters
        ----------
        run_id
            Run identifier to fetch steps for.

        Returns
        -------
        list[PipelineStepRecord]
            List of step records ordered by module, stage, and name.
        """
        cur = self.con.execute(
            """
            SELECT
                run_id,
                module,
                stage,
                name,
                started_at,
                completed_at,
                status,
                row_counts,
                extra
            FROM metadata.pipeline_steps
            WHERE run_id = ?
            ORDER BY module, stage, name
            """,
            [run_id],
        )

        rows = cur.fetchall()
        results: list[PipelineStepRecord] = []
        for (
            run_id_val,
            module,
            stage,
            name,
            started_at,
            completed_at,
            status,
            row_counts_raw,
            extra_raw,
        ) in rows:
            row_counts = json.loads(row_counts_raw) if row_counts_raw else None
            extra = json.loads(extra_raw) if extra_raw else None
            results.append(
                PipelineStepRecord(
                    run_id=str(run_id_val),
                    module=module,
                    stage=str(stage),
                    name=str(name),
                    status=status,
                    started_at=started_at,
                    completed_at=completed_at,
                    row_counts=row_counts,
                    extra=extra,
                )
            )
        return results

    def start_step(
        self,
        *,
        run_id: str,
        module: ModuleKind,
        stage: str,
        name: str,
    ) -> datetime:
        """Record the start of a pipeline step.

        Convenience method that creates a step record with status='running'.

        Parameters
        ----------
        run_id
            Parent run identifier.
        module
            Module executing the step.
        stage
            Stage within the module.
        name
            Step name (typically plugin name).

        Returns
        -------
        datetime
            The started_at timestamp for use in complete_step.
        """
        started_at = utc_now()
        record = PipelineStepRecord(
            run_id=run_id,
            module=module,
            stage=stage,
            name=name,
            status="running",
            started_at=started_at,
            completed_at=None,
            row_counts=None,
            extra=None,
        )
        self.record_step(record)
        return started_at

    def complete_step(self, params: StepCompletionParams) -> None:
        """Record the completion of a pipeline step.

        Convenience method that updates a step record with final status.

        Parameters
        ----------
        params
            Bundled completion parameters.
        """
        self.record_step(params.to_record())

    def fetch_recent_runs(self, *, limit: int = 10) -> list[PipelineRunRecord]:
        """Fetch the most recent pipeline runs.

        Parameters
        ----------
        limit
            Maximum number of runs to return.

        Returns
        -------
        list[PipelineRunRecord]
            List of run records ordered by started_at descending.
        """
        cur = self.con.execute(
            """
            SELECT
                run_id,
                repo,
                commit,
                kind,
                trigger,
                requested_operation,
                requested_datasets,
                started_at,
                completed_at,
                status,
                error_summary,
                pipeline_name
            FROM metadata.pipeline_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [limit],
        )
        rows = cur.fetchall()
        results: list[PipelineRunRecord] = []
        for (
            run_id_val,
            repo,
            commit,
            kind,
            trigger,
            requested_operation,
            requested_datasets_raw,
            started_at,
            completed_at,
            status,
            error_summary,
            pipeline_name,
        ) in rows:
            results.append(
                PipelineRunRecord(
                    run_id=str(run_id_val),
                    repo=str(repo),
                    commit=str(commit),
                    kind=str(kind),
                    trigger=str(trigger),
                    status=status,
                    started_at=started_at,
                    completed_at=completed_at,
                    requested_operation=str(requested_operation) if requested_operation else None,
                    requested_datasets=deserialize_str_tuple(requested_datasets_raw),
                    error_summary=str(error_summary) if error_summary else None,
                    pipeline_name=str(pipeline_name) if pipeline_name else None,
                )
            )
        return results


__all__ = [
    "ModuleKind",
    "PipelineRunRecord",
    "PipelineRunTracking",
    "PipelineStatus",
    "PipelineStepRecord",
    "StepStatus",
]
