"""Pipeline run tracking persistence for DuckDB.

This module provides persistent tracking of pipeline runs and their steps,
enabling observability and debugging of ingestion, graphs, analytics, export, and views
executions.

All DuckDB access is encapsulated here, following the storage layer pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
from sqlglot import exp

from codeintel.core.columnar.conversion import (
    reader_to_table,
    table_to_reader,
    tabular_to_arrow_reader,
)
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table, finalize_table
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.plan_ops import ScanPlanOptions, build_scan_plan
from codeintel.core.columnar.streaming import sample_reader
from codeintel.core.gateway import PipelineStepRecordProtocol
from codeintel.core.serialization.json import (
    decode_json_dict,
    deserialize_str_tuple,
    serialize_str_sequence,
)
from codeintel.core.serialization.payload import encode_payload
from codeintel.core.sqlglot_tools import render_sql_duckdb, table_expr_from_ref
from codeintel.core.time import utc_now
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.datasets.manifest_index import dataset_for_entry
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.query_results import (
    coerce_datetime,
    coerce_int,
    coerce_literal,
    coerce_optional_datetime,
    coerce_optional_str,
    coerce_str,
    iter_tuples_from_arrow_reader,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from duckdb import DuckDBPyConnection

    from codeintel.core.columnar.expr_vocab import Expression
    from codeintel.core.execution import RunContext
    from codeintel.storage.datasets import DatasetRegistry
    from codeintel.storage.datasets.manifest_index import DatasetManifestEntry
    from codeintel.storage.gateway.config import StorageConfig

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

_PIPELINE_RUNS_TABLE = meta_table_ref("metadata.pipeline_runs")
_PIPELINE_STEPS_TABLE = meta_table_ref("metadata.pipeline_steps")


def _arrow_scan_table(
    *,
    entry: DatasetManifestEntry,
    columns: list[str],
    filter_expr: Expression | None,
    order_by: Sequence[SortKey] | None,
    limit: int | None,
) -> pa.Table:
    dataset = dataset_for_entry(entry)
    plan = build_scan_plan(
        dataset,
        options=ScanPlanOptions(
            columns=columns,
            filter_expr=filter_expr,
            implicit_ordering=True,
            require_sequenced_output=True,
            order_by=order_by,
        ),
    )
    reader = plan.to_reader(use_threads=True)
    if limit is not None:
        reader = sample_reader(reader, max_rows=limit)
    table = reader_to_table(reader)
    finalized = finalize_table(
        table,
        spec=finalize_spec_for_table(entry.manifest.table_key, mode="tolerant"),
    )
    return finalized.good


def _coerce_row_counts(raw: dict[str, object]) -> dict[str, int]:
    """Coerce row count payload values to ints.

    Returns
    -------
    dict[str, int]
        Row counts normalized to integer values.
    """
    return {key: coerce_int(value, ctx=f"row_counts[{key}]") for key, value in raw.items()}


def _run_select_exprs() -> list[exp.Expression]:
    return [
        exp.Column(this=exp.to_identifier("run_id")),
        exp.Column(this=exp.to_identifier("repo")),
        exp.Column(this=exp.to_identifier("commit")),
        exp.Column(this=exp.to_identifier("kind")),
        exp.Column(this=exp.to_identifier("trigger")),
        exp.Column(this=exp.to_identifier("requested_operation")),
        exp.Column(this=exp.to_identifier("requested_datasets")),
        exp.Column(this=exp.to_identifier("started_at")),
        exp.Column(this=exp.to_identifier("completed_at")),
        exp.Column(this=exp.to_identifier("status")),
        exp.Column(this=exp.to_identifier("error_summary")),
        exp.Column(this=exp.to_identifier("pipeline_name")),
    ]


def _step_select_exprs() -> list[exp.Expression]:
    return [
        exp.Column(this=exp.to_identifier("run_id")),
        exp.Column(this=exp.to_identifier("module")),
        exp.Column(this=exp.to_identifier("stage")),
        exp.Column(this=exp.to_identifier("name")),
        exp.Column(this=exp.to_identifier("started_at")),
        exp.Column(this=exp.to_identifier("completed_at")),
        exp.Column(this=exp.to_identifier("status")),
        exp.Column(this=exp.to_identifier("row_counts")),
        exp.Column(this=exp.to_identifier("extra")),
    ]


def _build_upsert_insert(
    table_ref: str,
    *,
    columns: Sequence[str],
    conflict_columns: Sequence[str],
) -> exp.Insert:
    insert = exp.Insert(
        this=exp.Schema(
            this=table_expr_from_ref(table_ref),
            expressions=[exp.to_identifier(column) for column in columns],
        ),
        expression=exp.Values(
            expressions=[exp.Tuple(expressions=[exp.Placeholder() for _ in columns])]
        ),
    )
    update_columns = [column for column in columns if column not in conflict_columns]
    conflict_keys = [exp.to_identifier(column) for column in conflict_columns]
    if update_columns:
        assignments = [
            exp.EQ(
                this=exp.Column(this=exp.to_identifier(column)),
                expression=exp.Column(
                    this=exp.to_identifier(column),
                    table=exp.to_identifier("excluded"),
                ),
            )
            for column in update_columns
        ]
        conflict = exp.OnConflict(
            conflict_keys=conflict_keys,
            action=exp.Var(this="DO UPDATE"),
            expressions=assignments,
        )
    else:
        conflict = exp.OnConflict(
            conflict_keys=conflict_keys,
            action=exp.Var(this="DO NOTHING"),
        )
    insert.set("conflict", conflict)
    return insert


ModuleKind = Literal["ingestion", "graphs", "analytics", "export", "views", "build"]
"""Classification of pipeline module."""

_PIPELINE_STATUS_VALUES: tuple[PipelineStatus, ...] = (
    "running",
    "succeeded",
    "failed",
    "partial",
)
_STEP_STATUS_VALUES: tuple[StepStatus, ...] = (
    "pending",
    "running",
    "succeeded",
    "failed",
    "skipped",
)
_MODULE_KIND_VALUES: tuple[ModuleKind, ...] = (
    "ingestion",
    "graphs",
    "analytics",
    "export",
    "views",
    "build",
)


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
        Module that executed this step (ingestion, graphs, analytics, export, views).
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
    datasets: DatasetRegistry | None = None
    config: StorageConfig | None = None

    def _manifest_entry_for_table(self, table_key: str) -> DatasetManifestEntry | None:
        if self.datasets is None or self.config is None:
            return None
        snapshot_id = self.config.commit
        if snapshot_id is None:
            return None
        return self.datasets.manifest_entry_for_table(table_key, snapshot_id=snapshot_id)

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
        columns = [
            "run_id",
            "repo",
            "commit",
            "kind",
            "trigger",
            "requested_operation",
            "requested_datasets",
            "started_at",
            "completed_at",
            "status",
            "error_summary",
            "pipeline_name",
        ]
        insert_expr = _build_upsert_insert(
            _PIPELINE_RUNS_TABLE,
            columns=columns,
            conflict_columns=["run_id"],
        )
        self.con.execute(
            render_sql_duckdb(insert_expr),
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
        update_expr = exp.Update(
            this=table_expr_from_ref(_PIPELINE_RUNS_TABLE),
            expressions=[
                exp.EQ(this=exp.to_identifier("status"), expression=exp.Placeholder()),
                exp.EQ(
                    this=exp.to_identifier("error_summary"),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.to_identifier("completed_at"),
                    expression=exp.Placeholder(),
                ),
            ],
            where=exp.Where(
                this=exp.EQ(
                    this=exp.Column(this=exp.to_identifier("run_id")),
                    expression=exp.Placeholder(),
                )
            ),
        )
        self.con.execute(
            render_sql_duckdb(update_expr),
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
        query = (
            exp.select(*_run_select_exprs())
            .from_(table_expr_from_ref(_PIPELINE_RUNS_TABLE))
            .where(
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("run_id")),
                    expression=exp.Placeholder(),
                )
            )
        )
        cur = self.con.execute(render_sql_duckdb(query), [run_id])
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

    def record_step(self, record: PipelineStepRecordProtocol) -> None:
        """Insert or replace a pipeline step record.

        Can be called once at the end of a step, or twice (at start with
        status='running' and at end with final status).

        Parameters
        ----------
        record
            Step record to persist.
        """
        row_counts_payload = encode_payload(record.row_counts) if record.row_counts else None
        extra_payload = encode_payload(record.extra) if record.extra else None

        columns = [
            "run_id",
            "module",
            "stage",
            "name",
            "started_at",
            "completed_at",
            "status",
            "row_counts",
            "extra",
        ]
        insert_expr = _build_upsert_insert(
            _PIPELINE_STEPS_TABLE,
            columns=columns,
            conflict_columns=["run_id", "module", "name"],
        )
        self.con.execute(
            render_sql_duckdb(insert_expr),
            [
                record.run_id,
                record.module,
                record.stage,
                record.name,
                record.started_at,
                record.completed_at,
                record.status,
                row_counts_payload,
                extra_payload,
            ],
        )

    def fetch_steps(self, run_id: str) -> Sequence[PipelineStepRecordProtocol]:
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
        entry = self._manifest_entry_for_table(_PIPELINE_STEPS_TABLE)
        if entry is not None:
            columns = [
                "run_id",
                "module",
                "stage",
                "name",
                "started_at",
                "completed_at",
                "status",
                "row_counts",
                "extra",
            ]
            table = _arrow_scan_table(
                entry=entry,
                columns=columns,
                filter_expr=E.field("run_id") == E.scalar(run_id),
                order_by=[
                    ("module", "ascending"),
                    ("stage", "ascending"),
                    ("name", "ascending"),
                ],
                limit=None,
            )
            reader = table_to_reader(table, batch_size=DEFAULT_ARROW_BATCH_SIZE)
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
            ) in iter_tuples_from_arrow_reader(reader):
                row_counts = (
                    _coerce_row_counts(decode_json_dict(row_counts_raw)) if row_counts_raw else None
                )
                extra = decode_json_dict(extra_raw) if extra_raw else None
                results.append(
                    PipelineStepRecord(
                        run_id=coerce_str(run_id_val, ctx="pipeline_steps.run_id"),
                        module=coerce_literal(
                            module,
                            ctx="pipeline_steps.module",
                            allowed=_MODULE_KIND_VALUES,
                        ),
                        stage=coerce_str(stage, ctx="pipeline_steps.stage"),
                        name=coerce_str(name, ctx="pipeline_steps.name"),
                        status=coerce_literal(
                            status,
                            ctx="pipeline_steps.status",
                            allowed=_STEP_STATUS_VALUES,
                        ),
                        started_at=coerce_datetime(started_at, ctx="pipeline_steps.started_at"),
                        completed_at=coerce_optional_datetime(
                            completed_at,
                            ctx="pipeline_steps.completed_at",
                        ),
                        row_counts=row_counts,
                        extra=extra,
                    )
                )
            return results
        query = (
            exp.select(*_step_select_exprs())
            .from_(table_expr_from_ref(_PIPELINE_STEPS_TABLE))
            .where(
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("run_id")),
                    expression=exp.Placeholder(),
                )
            )
            .order_by(
                exp.Ordered(this=exp.Column(this=exp.to_identifier("module"))),
                exp.Ordered(this=exp.Column(this=exp.to_identifier("stage"))),
                exp.Ordered(this=exp.Column(this=exp.to_identifier("name"))),
            )
        )
        reader = tabular_to_arrow_reader(
            self.con.execute(
                render_sql_duckdb(query),
                [run_id],
            ),
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        )
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
        ) in iter_tuples_from_arrow_reader(reader):
            row_counts = (
                _coerce_row_counts(decode_json_dict(row_counts_raw)) if row_counts_raw else None
            )
            extra = decode_json_dict(extra_raw) if extra_raw else None
            results.append(
                PipelineStepRecord(
                    run_id=coerce_str(run_id_val, ctx="pipeline_steps.run_id"),
                    module=coerce_literal(
                        module,
                        ctx="pipeline_steps.module",
                        allowed=_MODULE_KIND_VALUES,
                    ),
                    stage=coerce_str(stage, ctx="pipeline_steps.stage"),
                    name=coerce_str(name, ctx="pipeline_steps.name"),
                    status=coerce_literal(
                        status,
                        ctx="pipeline_steps.status",
                        allowed=_STEP_STATUS_VALUES,
                    ),
                    started_at=coerce_datetime(started_at, ctx="pipeline_steps.started_at"),
                    completed_at=coerce_optional_datetime(
                        completed_at,
                        ctx="pipeline_steps.completed_at",
                    ),
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
        entry = self._manifest_entry_for_table(_PIPELINE_RUNS_TABLE)
        if entry is not None:
            columns = [
                "run_id",
                "repo",
                "commit",
                "kind",
                "trigger",
                "requested_operation",
                "requested_datasets",
                "started_at",
                "completed_at",
                "status",
                "error_summary",
                "pipeline_name",
            ]
            table = _arrow_scan_table(
                entry=entry,
                columns=columns,
                filter_expr=None,
                order_by=[("started_at", "descending")],
                limit=limit,
            )
            reader = table_to_reader(table, batch_size=DEFAULT_ARROW_BATCH_SIZE)
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
            ) in iter_tuples_from_arrow_reader(reader):
                results.append(
                    PipelineRunRecord(
                        run_id=coerce_str(run_id_val, ctx="pipeline_runs.run_id"),
                        repo=coerce_str(repo, ctx="pipeline_runs.repo"),
                        commit=coerce_str(commit, ctx="pipeline_runs.commit"),
                        kind=coerce_str(kind, ctx="pipeline_runs.kind"),
                        trigger=coerce_str(trigger, ctx="pipeline_runs.trigger"),
                        status=coerce_literal(
                            status,
                            ctx="pipeline_runs.status",
                            allowed=_PIPELINE_STATUS_VALUES,
                        ),
                        started_at=coerce_datetime(started_at, ctx="pipeline_runs.started_at"),
                        completed_at=coerce_optional_datetime(
                            completed_at,
                            ctx="pipeline_runs.completed_at",
                        ),
                        requested_operation=coerce_optional_str(
                            requested_operation,
                            ctx="pipeline_runs.requested_operation",
                        ),
                        requested_datasets=deserialize_str_tuple(requested_datasets_raw),
                        error_summary=coerce_optional_str(
                            error_summary,
                            ctx="pipeline_runs.error_summary",
                        ),
                        pipeline_name=coerce_optional_str(
                            pipeline_name,
                            ctx="pipeline_runs.pipeline_name",
                        ),
                    )
                )
            return results
        query = (
            exp.select(*_run_select_exprs())
            .from_(table_expr_from_ref(_PIPELINE_RUNS_TABLE))
            .order_by(exp.Ordered(this=exp.Column(this=exp.to_identifier("started_at")), desc=True))
            .limit(exp.Placeholder())
        )
        reader = tabular_to_arrow_reader(
            self.con.execute(
                render_sql_duckdb(query),
                [limit],
            ),
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        )
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
        ) in iter_tuples_from_arrow_reader(reader):
            results.append(
                PipelineRunRecord(
                    run_id=coerce_str(run_id_val, ctx="pipeline_runs.run_id"),
                    repo=coerce_str(repo, ctx="pipeline_runs.repo"),
                    commit=coerce_str(commit, ctx="pipeline_runs.commit"),
                    kind=coerce_str(kind, ctx="pipeline_runs.kind"),
                    trigger=coerce_str(trigger, ctx="pipeline_runs.trigger"),
                    status=coerce_literal(
                        status,
                        ctx="pipeline_runs.status",
                        allowed=_PIPELINE_STATUS_VALUES,
                    ),
                    started_at=coerce_datetime(started_at, ctx="pipeline_runs.started_at"),
                    completed_at=coerce_optional_datetime(
                        completed_at,
                        ctx="pipeline_runs.completed_at",
                    ),
                    requested_operation=coerce_optional_str(
                        requested_operation,
                        ctx="pipeline_runs.requested_operation",
                    ),
                    requested_datasets=deserialize_str_tuple(requested_datasets_raw),
                    error_summary=coerce_optional_str(
                        error_summary,
                        ctx="pipeline_runs.error_summary",
                    ),
                    pipeline_name=coerce_optional_str(
                        pipeline_name,
                        ctx="pipeline_runs.pipeline_name",
                    ),
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
