"""Kernel-lane helpers for row-changing operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import pyarrow as pa

from codeintel.core import columnar
from codeintel.core.columnar.execution_context import ExecutionContext, resolve_execution_context
from codeintel.core.columnar.explode_ops import (
    ExplodeResult,
    ExplodeSpec,
)
from codeintel.core.columnar.explode_ops import (
    explode_edges_for_join as _explode_edges_for_join,
)
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.finalize_ops import finalize_reader, finalize_spec_for_table
from codeintel.core.columnar.join_safe import join_safe_projection
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan
from codeintel.core.schemas.primitives import resolve_join_safe_columns

if TYPE_CHECKING:
    from codeintel.core.columnar.ordering import SortKey
    from codeintel.core.schemas.service import SchemaService

_DEFAULT_HASH_MODULUS = 2**63 - 1
_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"


class _StableDedupeFn(Protocol):
    def __call__(
        self,
        table: pa.Table,
        *,
        key_columns: Sequence[str],
        order_by: Sequence[SortKey] = (),
        tie_breakers: Sequence[SortKey] = (),
        require_tie_breakers: bool = False,
    ) -> pa.Table:
        """Return a stable dedupe result for a table."""


@dataclass(frozen=True, slots=True)
class GroupByMaxJoinBackSpec:
    """Specification for group-by max join-back operations."""

    key_columns: Sequence[str]
    score_column: str
    allowed_columns: Sequence[str] = ()
    table_key: str | None = None
    schema_service: SchemaService | None = None


@dataclass(frozen=True, slots=True)
class StableDedupeSpec:
    """Specification for stable dedupe with tie handling."""

    key_columns: Sequence[str]
    order_by: Sequence[SortKey] = ()
    tie_breakers: Sequence[SortKey] = ()
    require_tie_breakers: bool = False
    hash_tiebreaker: bool = False
    hash_columns: Sequence[str] | None = None
    hash_modulus: int | None = None


@dataclass(frozen=True, slots=True)
class WinnerSelectionSpec:
    """Specification for deterministic winner selection."""

    key_columns: Sequence[str]
    order_by: Sequence[SortKey] = ()
    tie_breakers: Sequence[SortKey] = ()
    require_tie_breakers: bool = False


@dataclass(frozen=True, slots=True)
class GroupedRollupSpec:
    """Specification for grouped rollup operations."""

    keys: Sequence[str] | None
    aggregates: Sequence[tuple[object, str, object | None, str]]
    pre_sort_keys: Sequence[SortKey] = ()
    order_by: Sequence[SortKey] = ()


def _resolve_join_safe_columns(
    *,
    allowed_columns: Sequence[str],
    table_key: str | None,
    schema_service: SchemaService | None,
) -> tuple[str, ...]:
    if allowed_columns:
        return tuple(allowed_columns)
    if table_key is None or schema_service is None:
        return ()
    table_schema = schema_service.get_table_schema(table_key)
    return resolve_join_safe_columns(table_schema)


def group_by_max_join_back(
    table: pa.Table,
    *,
    spec: GroupByMaxJoinBackSpec,
    ctx: ExecutionContext | None = None,
) -> pa.Table:
    """Select max-score rows per key via group-by and join-back.

    Parameters
    ----------
    table
        Input table with key and score columns.
    spec
        Group-by join-back specification.
    ctx
        Optional execution context for plan materialization.

    Returns
    -------
    pyarrow.Table
        Rows matching the max score per key.

    Raises
    ------
    ValueError
        If required key or score columns are missing.
    """
    if table.num_rows == 0:
        return table
    required = [*spec.key_columns, spec.score_column]
    missing = [name for name in required if name not in table.column_names]
    if missing:
        msg = f"group_by_max_join_back missing columns: {missing}"
        raise ValueError(msg)
    winners = (
        table.select(required)
        .group_by(list(spec.key_columns))
        .aggregate([(spec.score_column, "max")])
    )
    score_max_name = f"{spec.score_column}_max"
    resolved_allowed = _resolve_join_safe_columns(
        allowed_columns=spec.allowed_columns,
        table_key=spec.table_key,
        schema_service=spec.schema_service,
    )
    join_left = join_safe_projection(
        table,
        allowed_columns=(*spec.key_columns, spec.score_column, *resolved_allowed),
    )
    join_right = join_safe_projection(
        winners,
        allowed_columns=(*spec.key_columns, score_max_name),
    )
    join_spec = HashJoinSpec(
        left_keys=[*spec.key_columns, spec.score_column],
        right_keys=[*spec.key_columns, score_max_name],
        left_output=list(join_left.column_names),
        right_output=(),
    )
    plan = Plan.table(join_left).hash_join(right=Plan.table(join_right), spec=join_spec)
    execution_ctx = resolve_execution_context(ctx)
    return columnar.ExecutionPlan.from_plan(plan).to_table(ctx=execution_ctx)


def stable_dedupe_with_ties(
    table: pa.Table,
    *,
    spec: StableDedupeSpec,
) -> pa.Table:
    """Deduplicate rows using stable ordering and explicit tie handling.

    Parameters
    ----------
    table
        Input table to deduplicate.
    spec
        Stable dedupe specification with tie-breaker configuration.

    Returns
    -------
    pyarrow.Table
        Deduplicated table with tie handling applied.

    Raises
    ------
    ValueError
        If hash tie-breaking is enabled without hash columns.
    """
    if not spec.hash_tiebreaker or spec.tie_breakers:
        dedupe_fn = cast("_StableDedupeFn", columnar.stable_dedupe_with_ties)
        return dedupe_fn(
            table,
            key_columns=spec.key_columns,
            order_by=spec.order_by,
            tie_breakers=spec.tie_breakers,
            require_tie_breakers=spec.require_tie_breakers,
        )
    resolved_hash_columns = (
        tuple(spec.hash_columns) if spec.hash_columns is not None else tuple(spec.key_columns)
    )
    if not resolved_hash_columns:
        msg = "hash_tiebreaker requires hash_columns or key_columns"
        raise ValueError(msg)
    hash_name = _unique_column_name(table, base="_hash_tiebreaker")
    modulus = spec.hash_modulus if spec.hash_modulus is not None else _DEFAULT_HASH_MODULUS
    hash_values = columnar.hash_struct_ordinal(
        table,
        columns=resolved_hash_columns,
        modulus=modulus,
    )
    with_hash = table.append_column(hash_name, hash_values)
    dedupe_fn = cast("_StableDedupeFn", columnar.stable_dedupe_with_ties)
    deduped = dedupe_fn(
        with_hash,
        key_columns=spec.key_columns,
        order_by=spec.order_by,
        tie_breakers=((*spec.tie_breakers, (hash_name, "ascending"))),
        require_tie_breakers=spec.require_tie_breakers,
    )
    return deduped.drop([hash_name])


def select_winner_rows(
    table: pa.Table,
    *,
    spec: WinnerSelectionSpec,
) -> pa.Table:
    """Return a stable winner per key using explicit ordering rules.

    Parameters
    ----------
    table
        Input table containing candidate rows.
    spec
        Winner selection specification.

    Returns
    -------
    pyarrow.Table
        Table with a single winner per key.
    """
    return stable_dedupe_with_ties(
        table,
        spec=StableDedupeSpec(
            key_columns=spec.key_columns,
            order_by=spec.order_by,
            tie_breakers=spec.tie_breakers,
            require_tie_breakers=spec.require_tie_breakers,
        ),
    )


def explode_edges_for_join(
    table: pa.Table,
    *,
    spec: ExplodeSpec,
    allowed_columns: Sequence[str] = (),
    table_key: str | None = None,
    schema_service: SchemaService | None = None,
) -> ExplodeResult:
    """Explode list columns and return join-safe results.

    Parameters
    ----------
    table
        Table containing list payloads.
    spec
        Explode specification.
    allowed_columns
        Columns allowed to retain list payloads after explode.
    table_key
        Optional table key for schema-driven join-safe allowlists.
    schema_service
        Schema service used to resolve join-safe allowlists.

    Returns
    -------
    ExplodeResult
        Exploded rows with join-safe projections plus error rows.
    """
    resolved_allowed = _resolve_join_safe_columns(
        allowed_columns=allowed_columns,
        table_key=table_key,
        schema_service=schema_service,
    )
    return _explode_edges_for_join(
        table,
        spec=spec,
        allowed_columns=resolved_allowed,
    )


def grouped_rollup_table(
    table: pa.Table,
    *,
    spec: GroupedRollupSpec,
    ctx: ExecutionContext | None = None,
) -> pa.Table:
    """Return a grouped rollup table using plan lanes.

    Parameters
    ----------
    table
        Input table to aggregate.
    spec
        Grouped rollup specification.
    ctx
        Optional execution context for materialization.

    Returns
    -------
    pyarrow.Table
        Aggregated table with optional ordering applied.
    """
    plan = Plan.table(table)
    if spec.pre_sort_keys:
        plan = plan.order_by(sort_keys=spec.pre_sort_keys)
    plan = plan.aggregate(
        keys=[E.field(name) for name in spec.keys] if spec.keys else None,
        aggregates=spec.aggregates,
    )
    if spec.order_by:
        plan = plan.order_by(sort_keys=spec.order_by)
    execution_ctx = resolve_execution_context(ctx)
    reader = columnar.ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    result = finalize_reader(
        reader,
        spec=finalize_spec_for_table(
            _INTERNAL_PLAN_TABLE_KEY,
            mode="tolerant",
            ordering=plan.ordering,
        ),
    )
    return result.good


def build_grouped_rollup_plan(
    plan: Plan,
    *,
    keys: Sequence[str] | None,
    aggregates: Sequence[tuple[object, str, object | None, str]],
    order_by: Sequence[SortKey] = (),
) -> Plan:
    """Apply a group-by aggregate and optional ordering to a plan.

    Returns
    -------
    Plan
        Plan with aggregate (and optional order_by) applied.
    """
    key_exprs = [E.field(name) for name in keys] if keys else None
    plan = plan.aggregate(keys=key_exprs, aggregates=aggregates)
    if order_by:
        plan = plan.order_by(sort_keys=order_by)
    return plan


def _unique_column_name(table: pa.Table, *, base: str) -> str:
    if base not in table.column_names:
        return base
    suffix = 1
    while f"{base}_{suffix}" in table.column_names:
        suffix += 1
    return f"{base}_{suffix}"


__all__ = [
    "ExplodeResult",
    "ExplodeSpec",
    "GroupByMaxJoinBackSpec",
    "GroupedRollupSpec",
    "StableDedupeSpec",
    "WinnerSelectionSpec",
    "build_grouped_rollup_plan",
    "explode_edges_for_join",
    "group_by_max_join_back",
    "grouped_rollup_table",
    "select_winner_rows",
    "stable_dedupe_with_ties",
]
