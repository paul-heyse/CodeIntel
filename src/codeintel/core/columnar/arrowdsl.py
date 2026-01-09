"""Execution helpers for Arrow plans, tables, and finalize boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
from pyarrow import acero

from codeintel.core.columnar.finalize_ops import (
    FinalizeResult,
    FinalizeSpec,
    finalize_join_keys,
    finalize_table,
    record_join_precheck_errors,
)
from codeintel.core.columnar.kernels import SortKey, stable_sort_table
from codeintel.core.columnar.normalization import normalize_table_for_compute
from codeintel.core.validation.schema_constraints import is_list_like

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from codeintel.core.columnar.dedupe_ops import DedupeTier

    type TableThunk = Callable[[], pa.Table]
    type PostStep = Callable[[pa.Table], pa.Table]
else:
    type TableThunk = object
    type PostStep = object


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Execution context for Acero plans or table fallbacks."""

    use_threads: bool = True
    determinism: DedupeTier = "throughput"
    combine_chunks: bool = True


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

    inner: acero.Declaration | pa.Table | TableThunk

    def execute(self, *, ctx: ExecutionContext) -> pa.Table:
        """Materialize the plan into a table.

        Returns
        -------
        pyarrow.Table
            Materialized result.
        """
        if isinstance(self.inner, acero.Declaration):
            table = self.inner.to_table(use_threads=ctx.use_threads)
        elif isinstance(self.inner, pa.Table):
            table = self.inner
        else:
            table = self.inner()
        return normalize_table_for_compute(table, combine_chunks=ctx.combine_chunks)


def run_pipeline(
    *,
    plan: ExecutionPlan,
    post: Sequence[PostStep] = (),
    finalize: FinalizeSpec | PostStep | None = None,
    ctx: ExecutionContext | None = None,
) -> pa.Table:
    """Execute a plan, apply post steps, and optionally finalize.

    Returns
    -------
    pyarrow.Table
        Finalized output table.
    """
    resolved_ctx = ctx or ExecutionContext()
    table = plan.execute(ctx=resolved_ctx)
    for step in post:
        table = step(table)
    if finalize is None:
        return table
    if isinstance(finalize, FinalizeSpec):
        return finalize_table(table, spec=finalize).good
    return finalize(table)


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


def apply_deterministic_order(
    table: pa.Table,
    *,
    sort_keys: Sequence[SortKey] = (),
    determinism: DedupeTier = "throughput",
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
    if determinism == "canonical":
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
    "require_join_safe_schema",
    "run_pipeline",
]
