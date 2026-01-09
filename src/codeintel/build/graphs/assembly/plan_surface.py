"""Shared Acero plan surface for graph assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.tabular.expr_vocab import E, ExprVocab
from codeintel.build.tabular.plan_ops import HashJoinSpec, JoinType, Plan
from codeintel.core.columnar.ordering import SortKey

if TYPE_CHECKING:
    import pyarrow.dataset as ds


@dataclass(frozen=True, slots=True)
class GraphPlanSurface:
    """Plan helper surface for graph producers."""

    expr: ExprVocab = E

    def scan(
        self,
        dataset: ds.Dataset,
        *,
        columns: Sequence[str] | Mapping[str, pc.Expression] | None = None,
        filter_expr: pc.Expression | None = None,
        implicit_ordering: bool | None = None,
        require_sequenced_output: bool | None = None,
    ) -> Plan:
        """Create a scan plan for graph assembly."""
        return Plan.scan(
            dataset,
            columns=columns,
            filter_expr=filter_expr,
            implicit_ordering=implicit_ordering,
            require_sequenced_output=require_sequenced_output,
        )

    def table(self, table: pa.Table) -> Plan:
        """Create a plan from an in-memory table."""
        return Plan.table(table)

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
        """Build a hash join spec with defaults for graph pipelines."""
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
        """Apply an order_by node to the plan."""
        return plan.order_by(sort_keys=sort_keys, null_placement=null_placement)


graph_plan = GraphPlanSurface()

__all__ = ["GraphPlanSurface", "graph_plan"]
