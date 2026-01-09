"""Execution helpers for Arrow plans, tables, and finalize boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
from pyarrow import acero

from codeintel.core.columnar.dedupe_ops import DedupeTier, normalize_dedupe_tier
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.expr_vocab import E, Expression
from codeintel.core.columnar.finalize_ops import (
    FinalizeResult,
    FinalizeSpec,
    finalize_join_keys,
    finalize_reader,
    finalize_table,
    record_join_precheck_errors,
)
from codeintel.core.columnar.kernels import SortKey, stable_sort_table
from codeintel.core.columnar.normalization import normalize_table_for_compute
from codeintel.core.columnar.ordering import OrderingSpec
from codeintel.core.validation.schema_constraints import is_list_like

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    type TableThunk = Callable[[], pa.Table]
    type ReaderThunk = Callable[[], pa.RecordBatchReader]
    type PostStep = Callable[[pa.Table], pa.Table]
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
    """Plan wrapper for Acero declarations or table callables."""

    decl: acero.Declaration | None = None
    table_thunk: TableThunk | None = None
    reader_thunk: ReaderThunk | None = None
    ordering: OrderingSpec | None = None

    @classmethod
    def from_declaration(
        cls,
        declaration: acero.Declaration,
        *,
        ordering: OrderingSpec | None = None,
    ) -> ExecutionPlan:
        """Return an execution plan backed by an Acero declaration.

        Returns
        -------
        ExecutionPlan
            Plan wrapping the provided Acero declaration.
        """
        return cls(decl=declaration, ordering=ordering)

    @classmethod
    def from_table(
        cls,
        table: pa.Table,
        *,
        ordering: OrderingSpec | None = None,
    ) -> ExecutionPlan:
        """Return an execution plan backed by an in-memory table.

        Returns
        -------
        ExecutionPlan
            Plan wrapping the provided table.
        """
        return cls(table_thunk=lambda: table, ordering=ordering)

    @classmethod
    def from_reader(
        cls,
        reader: pa.RecordBatchReader,
        *,
        ordering: OrderingSpec | None = None,
    ) -> ExecutionPlan:
        """Return an execution plan backed by a record batch reader.

        Returns
        -------
        ExecutionPlan
            Plan wrapping the provided reader.
        """
        return cls(reader_thunk=lambda: reader, ordering=ordering)

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
        if self.decl is not None:
            return self.decl.to_reader(use_threads=ctx.use_threads)
        if self.reader_thunk is not None:
            return self.reader_thunk()
        if self.table_thunk is not None:
            return self.table_thunk().to_reader()
        msg = "ExecutionPlan requires a declaration, reader, or table thunk."
        raise RuntimeError(msg)

    def to_table(self, *, ctx: ExecutionContext) -> pa.Table:
        """Materialize the plan into a normalized table.

        Returns
        -------
        pyarrow.Table
            Normalized table for the plan output.
        """
        if self.table_thunk is not None:
            table = self.table_thunk()
        else:
            reader = self.to_reader(ctx=ctx)
            table = reader.read_all()
        return normalize_table_for_compute(table, combine_chunks=ctx.combine_chunks)


def run_pipeline(
    *,
    plan: ExecutionPlan,
    post: Sequence[PostStep] = (),
    finalize: FinalizeSpec,
    ctx: ExecutionContext | None = None,
) -> FinalizeResult:
    """Execute a plan, apply post steps, and finalize.

    Returns
    -------
    FinalizeResult
        Finalized result containing good rows, errors, and artifacts.
    """
    resolved_ctx = ctx or ExecutionContext()
    if post:
        table = plan.to_table(ctx=resolved_ctx)
        for step in post:
            table = step(table)
        return finalize_table(table, spec=finalize)
    reader = plan.to_reader(ctx=resolved_ctx)
    return finalize_reader(reader, spec=finalize)


def run_pipeline_good(
    *,
    plan: ExecutionPlan,
    post: Sequence[PostStep] = (),
    finalize: FinalizeSpec,
    ctx: ExecutionContext | None = None,
) -> pa.Table:
    """Execute a plan and return only the finalized good rows.

    Returns
    -------
    pyarrow.Table
        Finalized table of good rows.
    """
    return run_pipeline(plan=plan, post=post, finalize=finalize, ctx=ctx).good


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


def list_payload_columns(table: pa.Table) -> tuple[str, ...]:
    """Return column names containing list-like payloads.

    Parameters
    ----------
    table
        Table to inspect for list payloads.

    Returns
    -------
    tuple[str, ...]
        Column names containing list-like Arrow types.
    """
    return tuple(field.name for field in table.schema if is_list_like(field.type))


def require_join_safe_schema(
    table: pa.Table,
    *,
    allowed_columns: Sequence[str] = (),
) -> None:
    """Raise when list payloads are present in join inputs.

    Parameters
    ----------
    table
        Table to validate for join safety.
    allowed_columns
        Column names allowed to contain list payloads.

    Raises
    ------
    ValueError
        Raised when list payloads remain in disallowed columns.
    """
    allowed = set(allowed_columns)
    list_columns = [
        field.name
        for field in table.schema
        if is_list_like(field.type) and field.name not in allowed
    ]
    if not list_columns:
        return
    msg = f"Join inputs contain list payload columns: {list_columns}"
    raise ValueError(msg)


def join_safe_projection(
    table: pa.Table,
    *,
    allowed_columns: Sequence[str] = (),
) -> pa.Table:
    """Return a table projected to join-safe columns.

    Parameters
    ----------
    table
        Input table to project.
    allowed_columns
        Columns to retain when explicitly provided.

    Returns
    -------
    pyarrow.Table
        Join-safe projection of the input table.

    Raises
    ------
    ValueError
        Raised when projection removes all columns.
    """
    if allowed_columns:
        allowed_set = set(allowed_columns)
        keep = [name for name in table.column_names if name in allowed_set]
    else:
        keep = [field.name for field in table.schema if not is_list_like(field.type)]
    if not keep:
        msg = "Join-safe projection removed all columns; explode or whitelist columns."
        raise ValueError(msg)
    if keep == list(table.column_names):
        return table
    return table.select(keep)


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
    "apply_deterministic_order",
    "join_safe_projection",
    "list_payload_columns",
    "precheck_join_keys",
    "project_struct_fields",
    "require_join_safe_schema",
    "run_pipeline",
    "run_pipeline_good",
]
