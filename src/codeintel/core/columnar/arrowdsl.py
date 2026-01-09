"""Execution helpers for Arrow plans, tables, and finalize boundaries."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
from pyarrow import acero

from codeintel.core.columnar.dedupe_ops import DedupeTier, normalize_dedupe_tier
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_execution_context,
)
from codeintel.core.columnar.expr_vocab import E, Expression
from codeintel.core.columnar.finalize_ops import (
    FinalizeResult,
    FinalizeSpec,
    finalize_join_keys,
    finalize_reader,
    finalize_table,
    record_join_precheck_errors,
    resolve_finalize_spec,
)
from codeintel.core.columnar.join_safe import (
    join_safe_projection,
    list_payload_columns,
    require_join_safe_schema,
)
from codeintel.core.columnar.kernels import SortKey, stable_sort_table
from codeintel.core.columnar.normalization import normalize_table_for_compute
from codeintel.core.columnar.ordering import OrderingSpec
from codeintel.core.columnar.plan_ops import ExternalPlanRequest, Plan, run_external_plan
from codeintel.core.columnar.run_manifest import (
    RunManifestOptions,
    run_manifest_options_for_context,
    write_run_manifest,
)
from codeintel.core.columnar.streaming import configure_arrow_threading_for_context

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    type TableThunk = Callable[[], pa.Table]
    type ReaderThunk = Callable[[], pa.RecordBatchReader]
    type PostStep = Callable[[pa.Table], pa.Table]

    from codeintel.core.columnar.streaming import ScanTelemetry
else:
    type TableThunk = object
    type ReaderThunk = object
    type PostStep = object


@dataclass(frozen=True, slots=True)
class JoinPrecheckSpec:
    """Specification for join-key precheck evaluation."""

    required_non_null: Sequence[str]
    key_fields: Sequence[str] = ()
    context_fields: Sequence[str] = ()
    table_key: str | None = None
    target_name: str | None = None
    stage: str = "schema"
    record: bool = True


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Plan wrapper for Acero declarations, external plans, or table callables."""

    decl: acero.Declaration | None = None
    table_thunk: TableThunk | None = None
    reader_thunk: ReaderThunk | None = None
    external_request: ExternalPlanRequest | None = None
    ordering: OrderingSpec | None = None
    determinism: DedupeTier | None = None

    @classmethod
    def from_declaration(
        cls,
        declaration: acero.Declaration,
        *,
        ordering: OrderingSpec | None = None,
        determinism: DedupeTier | None = None,
    ) -> ExecutionPlan:
        """Return an execution plan backed by an Acero declaration.

        Returns
        -------
        ExecutionPlan
            Plan wrapping the provided Acero declaration.
        """
        return cls(decl=declaration, ordering=ordering, determinism=determinism)

    @classmethod
    def from_table(
        cls,
        table: pa.Table,
        *,
        ordering: OrderingSpec | None = None,
        determinism: DedupeTier | None = None,
    ) -> ExecutionPlan:
        """Return an execution plan backed by an in-memory table.

        Returns
        -------
        ExecutionPlan
            Plan wrapping the provided table.
        """
        return cls(table_thunk=lambda: table, ordering=ordering, determinism=determinism)

    @classmethod
    def from_reader(
        cls,
        reader: pa.RecordBatchReader,
        *,
        ordering: OrderingSpec | None = None,
        determinism: DedupeTier | None = None,
    ) -> ExecutionPlan:
        """Return an execution plan backed by a record batch reader.

        Returns
        -------
        ExecutionPlan
            Plan wrapping the provided reader.
        """
        return cls(reader_thunk=lambda: reader, ordering=ordering, determinism=determinism)

    @classmethod
    def from_external_plan(
        cls,
        request: ExternalPlanRequest,
        *,
        ordering: OrderingSpec | None = None,
        determinism: DedupeTier | None = None,
    ) -> ExecutionPlan:
        """Return an execution plan backed by an external plan request.

        Returns
        -------
        ExecutionPlan
            Plan wrapping the external plan runner request.
        """
        return cls(external_request=request, ordering=ordering, determinism=determinism)

    @classmethod
    def from_plan(
        cls,
        plan: Plan,
        *,
        determinism: DedupeTier | None = None,
    ) -> ExecutionPlan:
        """Return an execution plan backed by an Acero Plan.

        Returns
        -------
        ExecutionPlan
            Plan wrapper retaining ordering metadata from the DSL plan.
        """
        return cls(
            decl=plan.declaration,
            ordering=plan.ordering,
            determinism=determinism,
        )

    def to_reader(self, *, ctx: ExecutionContext) -> pa.RecordBatchReader:
        """Return a RecordBatchReader for the plan output.

        Returns
        -------
        pyarrow.RecordBatchReader
            Reader yielding plan output batches.

        Raises
        ------
        RuntimeError
            Raised when no declaration, reader, or table thunk is available.
        """
        configure_arrow_threading_for_context(ctx=ctx)
        resolved_use_threads = ctx.resolve_use_threads()
        if self.decl is not None:
            return self.decl.to_reader(use_threads=resolved_use_threads)
        if self.external_request is not None:
            request = self.external_request
            if request.use_threads is None:
                request = replace(request, use_threads=resolved_use_threads)
            return run_external_plan(request)
        if self.reader_thunk is not None:
            return self.reader_thunk()
        if self.table_thunk is not None:
            return self.table_thunk().to_reader()
        msg = "ExecutionPlan requires a declaration, external plan, reader, or table thunk."
        raise RuntimeError(msg)

    def to_table(self, *, ctx: ExecutionContext) -> pa.Table:
        """Materialize the plan into a normalized table.

        Returns
        -------
        pyarrow.Table
            Normalized table for the plan output.
        """
        configure_arrow_threading_for_context(ctx=ctx)
        if self.table_thunk is not None:
            table = self.table_thunk()
        else:
            reader = self.to_reader(ctx=ctx)
            table = reader.read_all()
        return normalize_table_for_compute(table, combine_chunks=ctx.combine_chunks)


@dataclass(frozen=True, slots=True)
class PipelineRunOptions:
    """Optional settings for pipeline execution."""

    post: Sequence[PostStep] = ()
    ctx: ExecutionContext | None = None
    manifest_dir: Path | None = None
    manifest_options: RunManifestOptions | None = None
    scan_telemetry: ScanTelemetry | None = None


def run_pipeline(
    *,
    plan: ExecutionPlan,
    finalize: FinalizeSpec,
    options: PipelineRunOptions | None = None,
) -> FinalizeResult:
    """Execute a plan, apply post steps, and finalize.

    Returns
    -------
    FinalizeResult
        Finalized result containing good rows, errors, and artifacts.
    """
    resolved_options = options or PipelineRunOptions()
    start = time.monotonic()
    resolved_ctx = resolve_execution_context(resolved_options.ctx)
    finalize_spec = _resolve_finalize_spec(finalize, plan=plan, ctx=resolved_ctx)
    result, timings = _execute_plan_with_finalize(
        plan=plan,
        finalize_spec=finalize_spec,
        ctx=resolved_ctx,
        post_steps=resolved_options.post,
    )
    duration_seconds = time.monotonic() - start
    if resolved_options.manifest_dir is not None:
        resolved_options.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_options = resolved_options.manifest_options
        extras = {
            "duration_seconds": duration_seconds,
        }
        if manifest_options is None:
            manifest_options = RunManifestOptions(
                plan_seconds=timings["plan_seconds"],
                post_seconds=timings["post_seconds"],
                finalize_seconds=timings["finalize_seconds"],
                extras=extras,
            )
        else:
            merged_extras = {**(manifest_options.extras or {}), **extras}
            manifest_options = replace(
                manifest_options,
                extras=merged_extras,
                plan_seconds=timings["plan_seconds"],
                post_seconds=timings["post_seconds"],
                finalize_seconds=timings["finalize_seconds"],
            )
        resolved_manifest_options = run_manifest_options_for_context(
            ctx=resolved_ctx,
            ordering=finalize_spec.ordering or plan.ordering,
            scan_telemetry=resolved_options.scan_telemetry,
            options=manifest_options,
        )
        write_run_manifest(resolved_options.manifest_dir, options=resolved_manifest_options)
    return result


def _execute_plan_with_finalize(
    *,
    plan: ExecutionPlan,
    finalize_spec: FinalizeSpec,
    ctx: ExecutionContext,
    post_steps: Sequence[PostStep],
) -> tuple[FinalizeResult, dict[str, float]]:
    plan_start = time.monotonic()
    post_seconds = 0.0
    if post_steps:
        table = plan.to_table(ctx=ctx)
        plan_seconds = time.monotonic() - plan_start
        post_start = time.monotonic()
        for step in post_steps:
            table = step(table)
        post_seconds = time.monotonic() - post_start
        finalize_start = time.monotonic()
        result = finalize_table(table, spec=finalize_spec)
        finalize_seconds = time.monotonic() - finalize_start
    else:
        reader = plan.to_reader(ctx=ctx)
        plan_seconds = time.monotonic() - plan_start
        finalize_start = time.monotonic()
        result = finalize_reader(reader, spec=finalize_spec)
        finalize_seconds = time.monotonic() - finalize_start
    return result, {
        "plan_seconds": plan_seconds,
        "post_seconds": post_seconds,
        "finalize_seconds": finalize_seconds,
    }


def run_pipeline_good(
    *,
    plan: ExecutionPlan,
    finalize: FinalizeSpec,
    options: PipelineRunOptions | None = None,
) -> pa.Table:
    """Execute a plan and return only the finalized good rows.

    Returns
    -------
    pyarrow.Table
        Finalized table of good rows.
    """
    return run_pipeline(plan=plan, finalize=finalize, options=options).good


def _resolve_finalize_spec(
    spec: FinalizeSpec,
    *,
    plan: ExecutionPlan,
    ctx: ExecutionContext,
) -> FinalizeSpec:
    determinism = spec.determinism
    if determinism is None:
        determinism = plan.determinism or ctx.resolve_determinism()
    ordering = spec.ordering or plan.ordering
    if determinism == spec.determinism and ordering == spec.ordering:
        return resolve_finalize_spec(spec)
    return resolve_finalize_spec(
        replace(spec, determinism=determinism, ordering=ordering)
    )


def precheck_join_keys(
    table: pa.Table,
    *,
    spec: JoinPrecheckSpec,
) -> FinalizeResult:
    """Validate join keys and optionally record precheck errors.

    Parameters
    ----------
    table
        Table to validate.
    spec
        Join precheck specification for required fields and metadata.

    Returns
    -------
    FinalizeResult
        Result of the join-key precheck.
    """
    result = finalize_join_keys(
        table,
        required_non_null=spec.required_non_null,
        key_fields=spec.key_fields,
        context_fields=spec.context_fields,
        stage=spec.stage,
    )
    if spec.record:
        record_join_precheck_errors(
            result,
            table_key=spec.table_key,
            target_name=spec.target_name,
            join_keys=spec.required_non_null,
        )
    return result


def project_struct_fields(
    struct_column: str,
    fields: Sequence[str],
    *,
    prefix: str | None = None,
) -> dict[str, Expression]:
    """Return projection mapping for struct fields.

    Parameters
    ----------
    struct_column
        Name of the struct column to project from.
    fields
        Struct field names to project.
    prefix
        Optional prefix for projected column names.

    Returns
    -------
    dict[str, Expression]
        Mapping of output column names to struct field expressions.
    """
    output: dict[str, Expression] = {}
    for name in fields:
        output_name = f"{prefix}{name}" if prefix else name
        output[output_name] = E.field((struct_column, name))
    return output


def apply_deterministic_order(
    table: pa.Table,
    *,
    sort_keys: Sequence[SortKey] = (),
    determinism: DedupeTier = "stable_set",
) -> pa.Table:
    """Apply deterministic ordering for canonical deterministic tiers.

    Parameters
    ----------
    table
        Table to order.
    sort_keys
        Sort keys to apply for deterministic ordering.
    determinism
        Determinism tier controlling enforcement behavior.

    Returns
    -------
    pyarrow.Table
        Ordered table when sort keys are provided.

    Raises
    ------
    ValueError
        Raised when canonical determinism lacks sort keys.
    """
    if sort_keys:
        return stable_sort_table(table, sort_keys=sort_keys)
    if normalize_dedupe_tier(determinism) == "canonical":
        msg = "Canonical determinism requires sort_keys for stable ordering."
        raise ValueError(msg)
    return table


__all__ = [
    "ExecutionContext",
    "ExecutionPlan",
    "JoinPrecheckSpec",
    "PipelineRunOptions",
    "apply_deterministic_order",
    "join_safe_projection",
    "list_payload_columns",
    "precheck_join_keys",
    "project_struct_fields",
    "require_join_safe_schema",
    "run_pipeline",
    "run_pipeline_good",
]
