"""Shared Acero plan surface for graph assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

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
from codeintel.core.columnar.queryspec import QuerySpec, projection_spec_from_columns
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    import pyarrow.dataset as ds

    from codeintel.core.columnar.execution_context import ExecutionContext


@dataclass(frozen=True, slots=True)
class GraphPlanSurface:
    """Plan helper surface for graph producers."""

    expr: ExprVocab = E

    def scan(
        self,
        dataset: ds.Dataset,
        *,
        table_key: str | None = None,
        columns: Sequence[str] | Mapping[str, pc.Expression] | None = None,
        filter_expr: pc.Expression | None = None,
        implicit_ordering: bool | None = None,
        require_sequenced_output: bool | None = None,
        ctx: ExecutionContext | None = None,
    ) -> Plan:
        """Create a scan plan for graph assembly.

        Returns
        -------
        Plan
            Scan plan for graph assembly inputs.
        """
        options = QueryPlanOptions(
            implicit_ordering=implicit_ordering,
            require_sequenced_output=require_sequenced_output,
        )
        if table_key is not None:
            return plan_from_schema_defaults(
                schema_service=get_schema_service(),
                request=SchemaPlanDefaultsRequest(
                    table_key=table_key,
                    dataset=dataset,
                    predicate=filter_expr,
                    columns=columns,
                    options=options,
                    ctx=ctx,
                ),
            )
        projection = projection_spec_from_columns(
            columns,
            default_columns=tuple(dataset.schema.names),
        )
        query_spec = QuerySpec(
            predicate=filter_expr,
            pushdown_predicate=filter_expr,
            projection=projection,
        )
        return build_query_plan(dataset, spec=query_spec, options=options)

    def table(self, table: pa.Table) -> Plan:
        """Create a plan from an in-memory table.

        Returns
        -------
        Plan
            Plan backed by the provided table.
        """
        return build_table_plan(table=table)

    def hash_join_spec(
        self,
        *,
        how: JoinType,
        left_keys: Sequence[str],
        right_keys: Sequence[str] | None = None,
        left_output: Sequence[str] | None = None,
        right_output: Sequence[str] | None = None,
        output_suffix_for_left: str | None = None,
        output_suffix_for_right: str | None = None,
        filter_expression: pc.Expression | None = None,
    ) -> HashJoinSpec:
        """Build a hash join spec with defaults for graph pipelines.

        Returns
        -------
        HashJoinSpec
            Hash join spec with default graph pipeline settings.
        """
        return HashJoinSpec(
            left_keys=tuple(left_keys),
            right_keys=tuple(right_keys) if right_keys is not None else tuple(left_keys),
            how=how,
            left_output=tuple(left_output) if left_output is not None else None,
            right_output=tuple(right_output) if right_output is not None else None,
            output_suffix_for_left=output_suffix_for_left,
            output_suffix_for_right=output_suffix_for_right,
            filter_expression=filter_expression,
        )

    def order_by(
        self,
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

__all__ = ["GraphPlanSurface", "graph_plan"]
