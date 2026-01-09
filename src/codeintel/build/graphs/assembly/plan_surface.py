"""Shared Acero plan surface for graph assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.expr_vocab import E, ExprVocab
from codeintel.core.columnar.ordering import SortKey
from codeintel.core.columnar.plan_builder import (
    SchemaPlanDefaultsRequest,
    build_table_plan,
    plan_from_schema_defaults,
)
from codeintel.core.columnar.plan_ops import (
    HashJoinSpec,
    JoinType,
    Plan,
    QueryPlanOptions,
    build_query_plan,
)
from codeintel.core.columnar.queryspec import (
    PROVENANCE_FIELDS,
    QuerySpec,
    projection_spec_from_schema_defaults,
)
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    import pyarrow.dataset as ds

    from codeintel.core.columnar.execution_context import ExecutionContext


@dataclass(frozen=True, slots=True)
class ScanPlanRequest:
    """Scan request inputs for graph plan construction."""

    dataset: ds.Dataset
    table_key: str | None = None
    columns: Sequence[str] | Mapping[str, pc.Expression] | None = None
    filter_expr: pc.Expression | None = None
    implicit_ordering: bool | None = None
    require_sequenced_output: bool | None = None
    ctx: ExecutionContext | None = None


@dataclass(frozen=True, slots=True)
class HashJoinSpecRequest:
    """Hash join settings for graph plan construction."""

    how: JoinType
    left_keys: Sequence[str]
    right_keys: Sequence[str] | None = None
    left_output: Sequence[str] | None = None
    right_output: Sequence[str] | None = None
    output_suffix_for_left: str | None = None
    output_suffix_for_right: str | None = None
    filter_expression: pc.Expression | None = None


def _provenance_columns_for_request(
    *,
    ctx: ExecutionContext | None,
    available_columns: Sequence[str],
) -> tuple[str, ...]:
    resolved_ctx = resolve_execution_context(ctx)
    provenance = resolved_ctx.provenance
    profile = resolved_ctx.runtime_profile
    if profile is not None:
        provenance = profile.resolve_provenance(default=provenance)
    if resolved_ctx.resolve_determinism() == "canonical":
        provenance = True
    if not provenance:
        return ()
    available = set(available_columns)
    return tuple(
        output_name
        for output_name, _source_name in PROVENANCE_FIELDS
        if output_name in available
    )


@dataclass(frozen=True, slots=True)
class GraphPlanSurface:
    """Plan helper surface for graph producers."""

    expr: type[ExprVocab] = E

    @staticmethod
    def scan(request: ScanPlanRequest) -> Plan:
        """Create a scan plan for graph assembly.

        Returns
        -------
        Plan
            Scan plan for graph assembly inputs.
        """
        options = QueryPlanOptions(
            implicit_ordering=request.implicit_ordering,
            require_sequenced_output=request.require_sequenced_output,
        )
        if request.table_key is not None:
            return plan_from_schema_defaults(
                schema_service=get_schema_service(),
                request=SchemaPlanDefaultsRequest(
                    table_key=request.table_key,
                    dataset=request.dataset,
                    predicate=request.filter_expr,
                    columns=request.columns,
                    options=options,
                    ctx=request.ctx,
                ),
            )
        available_columns = tuple(request.dataset.schema.names)
        provenance_columns = _provenance_columns_for_request(
            ctx=request.ctx,
            available_columns=available_columns,
        )
        projection = projection_spec_from_schema_defaults(
            request.columns,
            table_schema=None,
            available_columns=available_columns,
            provenance_columns=provenance_columns,
        )
        query_spec = QuerySpec(
            predicate=request.filter_expr,
            pushdown_predicate=request.filter_expr,
            projection=projection,
        )
        return build_query_plan(request.dataset, spec=query_spec, options=options)

    @staticmethod
    def table(table: pa.Table) -> Plan:
        """Create a plan from an in-memory table.

        Returns
        -------
        Plan
            Plan backed by the provided table.
        """
        return build_table_plan(table=table)

    @staticmethod
    def hash_join_spec(request: HashJoinSpecRequest) -> HashJoinSpec:
        """Build a hash join spec with defaults for graph pipelines.

        Returns
        -------
        HashJoinSpec
            Hash join spec with default graph pipeline settings.
        """
        return HashJoinSpec(
            left_keys=tuple(request.left_keys),
            right_keys=tuple(request.right_keys)
            if request.right_keys is not None
            else tuple(request.left_keys),
            how=request.how,
            left_output=tuple(request.left_output) if request.left_output is not None else None,
            right_output=tuple(request.right_output) if request.right_output is not None else None,
            output_suffix_for_left=request.output_suffix_for_left,
            output_suffix_for_right=request.output_suffix_for_right,
            filter_expression=request.filter_expression,
        )

    @staticmethod
    def order_by(
        plan: Plan,
        *,
        sort_keys: Sequence[SortKey],
        null_placement: str = "at_end",
    ) -> Plan:
        """Apply an order_by node to the plan.

        Returns
        -------
        Plan
            Plan with an order_by node applied.
        """
        return plan.order_by(sort_keys=sort_keys, null_placement=null_placement)


graph_plan = GraphPlanSurface()

__all__ = [
    "GraphPlanSurface",
    "HashJoinSpecRequest",
    "ScanPlanRequest",
    "graph_plan",
]
