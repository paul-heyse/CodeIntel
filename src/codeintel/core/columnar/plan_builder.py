"""Plan construction helpers for Acero plan assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import (
    Plan,
    build_query_plan,
    query_plan_options_for_context,
)
from codeintel.core.columnar.queryspec import (
    PROVENANCE_FIELDS,
    ProjectionSpec,
    QuerySpec,
    projection_spec_from_schema_defaults,
)
from codeintel.core.schemas.primitives import resolve_canonical_sort_keys

if TYPE_CHECKING:
    import pyarrow.dataset as ds

    from codeintel.core.columnar.ordering import SortKey
    from codeintel.core.columnar.plan_ops import QueryPlanOptions
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.service import SchemaService


def require_columns(table: pa.Table, columns: Sequence[str]) -> None:
    """Require that the provided columns exist on a table.

    Raises
    ------
    ValueError
        If any required columns are missing from the table.
    """
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        msg = f"Missing snapshot columns: {missing}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TablePlanOptions:
    """Options for building table plans."""

    filter_expr: pc.Expression | None = None
    projection: Mapping[str, pc.Expression] | Sequence[pc.Expression] | None = None
    names: Sequence[str] | None = None
    order_by: Sequence[SortKey] = ()
    null_placement: str = "at_end"


@dataclass(frozen=True, slots=True)
class SchemaPlanDefaultsRequest:
    """Request for schema-driven plan defaults."""

    table_key: str
    dataset: ds.Dataset
    predicate: pc.Expression | None = None
    columns: Sequence[str] | Mapping[str, pc.Expression] | None = None
    options: QueryPlanOptions | None = None
    ctx: ExecutionContext | None = None


def build_snapshot_query_spec(
    *,
    base_cols: Sequence[str],
    repo: str | None = None,
    commit: str | None = None,
    computed: Sequence[tuple[str, pc.Expression]] = (),
    table: pa.Table | None = None,
) -> QuerySpec:
    """Build a QuerySpec scoped to a repo/commit snapshot.

    Returns
    -------
    QuerySpec
        Snapshot-scoped query specification with optional projection.
    """
    if table is not None:
        require_columns(table, base_cols)
        available = set(table.column_names)
    else:
        available = None
    predicate = _snapshot_predicate(
        available=available,
        repo=repo,
        commit=commit,
    )
    projection = ProjectionSpec(
        base_cols=tuple(base_cols),
        computed=tuple(computed),
    )
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )


def build_snapshot_plan(
    *,
    table: pa.Table,
    spec: QuerySpec,
    ctx: ExecutionContext | None = None,
) -> Plan:
    """Build a Plan scoped to a repo/commit snapshot.

    Parameters
    ----------
    table
        Input table to scope.
    spec
        QuerySpec describing predicate and projection.
    ctx
        Optional execution context to determine provenance inclusion.

    Returns
    -------
    Plan
        Plan filtered to the snapshot and optionally projected.
    """
    return build_plan_from_query_spec(table=table, spec=spec, ctx=ctx)


def build_plan_from_query_spec(
    *,
    table: pa.Table,
    spec: QuerySpec,
    ctx: ExecutionContext | None = None,
) -> Plan:
    """Build a Plan from a QuerySpec for an in-memory table.

    Returns
    -------
    Plan
        Plan filtered and projected per the query spec.
    """
    plan = Plan.table(table)
    if spec.predicate is not None:
        plan = plan.filter(spec.predicate)
    projection = spec.project_expressions(provenance=_include_provenance(table, ctx=ctx))
    if projection:
        plan = plan.project(projection)
    return plan


def build_table_plan(
    *,
    table: pa.Table,
    options: TablePlanOptions | None = None,
) -> Plan:
    """Build a Plan from an in-memory table with optional filtering and projection.

    Returns
    -------
    Plan
        Plan with filter/project/order_by nodes applied as requested.
    """
    resolved = options or TablePlanOptions()
    plan = Plan.table(table)
    if resolved.filter_expr is not None:
        plan = plan.filter(resolved.filter_expr)
    if resolved.projection is not None:
        plan = plan.project(resolved.projection, names=resolved.names)
    if resolved.order_by:
        plan = plan.order_by(
            sort_keys=resolved.order_by,
            null_placement=resolved.null_placement,
        )
    return plan


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


def plan_from_schema_defaults(
    *,
    schema_service: SchemaService,
    request: SchemaPlanDefaultsRequest,
) -> Plan:
    """Build a scan plan using schema-driven defaults.

    Returns
    -------
    Plan
        Plan compiled with schema defaults for projection and ordering.
    """
    table_schema = schema_service.get_table_schema(request.table_key)
    projection = projection_spec_from_schema_defaults(
        request.columns,
        table_schema=table_schema,
        available_columns=tuple(request.dataset.schema.names),
    )
    spec = QuerySpec(
        predicate=request.predicate,
        pushdown_predicate=request.predicate,
        projection=projection,
    )
    resolved_options = query_plan_options_for_context(
        ctx=request.ctx,
        options=request.options,
    )
    resolved_options = _apply_canonical_order_by(
        resolved_options,
        ctx=request.ctx,
        schema=table_schema,
    )
    return build_query_plan(request.dataset, spec=spec, options=resolved_options)


def _snapshot_predicate(
    *,
    available: set[str] | None,
    repo: str | None,
    commit: str | None,
) -> pc.Expression | None:
    filters: list[pc.Expression] = []
    if repo is not None and (available is None or "repo" in available):
        filters.append(E.field("repo") == E.scalar(repo))
    if commit is not None and (available is None or "commit" in available):
        filters.append(E.field("commit") == E.scalar(commit))
    if not filters:
        return None
    return E.and_(*filters)


def _apply_canonical_order_by(
    options: QueryPlanOptions,
    *,
    ctx: ExecutionContext | None,
    schema: TableSchema | None,
) -> QueryPlanOptions:
    if options.order_by is not None:
        return options
    if ctx is None or ctx.resolve_determinism() != "canonical":
        return options
    canonical_keys = resolve_canonical_sort_keys(schema)
    if not canonical_keys:
        return options
    direction = "ascending"
    order_by: tuple[SortKey, ...] = tuple((key, direction) for key in canonical_keys)
    return replace(options, order_by=order_by)


def _include_provenance(table: pa.Table, *, ctx: ExecutionContext | None) -> bool:
    if ctx is None:
        return False
    resolved = ctx.provenance
    if ctx.runtime_profile is not None:
        resolved = ctx.runtime_profile.resolve_provenance(default=resolved)
    if not resolved:
        return False
    column_names = set(table.column_names)
    return all(output_name in column_names for output_name, _source_name in PROVENANCE_FIELDS)


__all__ = [
    "SchemaPlanDefaultsRequest",
    "TablePlanOptions",
    "build_grouped_rollup_plan",
    "build_plan_from_query_spec",
    "build_snapshot_plan",
    "build_snapshot_query_spec",
    "build_table_plan",
    "plan_from_schema_defaults",
    "require_columns",
]
