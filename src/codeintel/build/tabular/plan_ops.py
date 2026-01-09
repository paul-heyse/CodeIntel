"""Acero plan helpers for Arrow-first pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import acero

from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.normalization import normalize_table_for_compute

if TYPE_CHECKING:
    import pyarrow.dataset as ds

JoinType = Literal[
    "left semi",
    "right semi",
    "left anti",
    "right anti",
    "inner",
    "left outer",
    "right outer",
    "full outer",
]


@dataclass(frozen=True, slots=True)
class HashJoinSpec:
    """Hash join configuration for plan joins."""

    left_keys: Sequence[str]
    right_keys: Sequence[str]
    how: JoinType = "left outer"
    left_output: Sequence[str] | None = None
    right_output: Sequence[str] | None = None
    output_suffix_for_left: str | None = None
    output_suffix_for_right: str | None = None
    filter_expression: pc.Expression | None = None


@dataclass(frozen=True, slots=True)
class Plan:
    """Wrap an Acero declaration for plan construction."""

    declaration: acero.Declaration
    schema: pa.Schema | None = None

    @classmethod
    def scan(
        cls,
        dataset: ds.Dataset,
        *,
        columns: Sequence[str] | Mapping[str, pc.Expression] | None = None,
        filter_expr: pc.Expression | None = None,
        implicit_ordering: bool | None = None,
        require_sequenced_output: bool | None = None,
    ) -> Plan:
        """Create a scan declaration for a dataset.

        Parameters
        ----------
        dataset
            Dataset to scan.
        columns
            Column names or projection expressions to push down.
        filter_expr
            Optional filter expression.
        implicit_ordering
            Whether to request implicit ordering for stable plans.
        require_sequenced_output
            Whether to require sequenced output batches.

        Returns
        -------
        Plan
            Plan seeded with a scan declaration.
        """
        kwargs: dict[str, object] = {}
        if columns is not None:
            kwargs["columns"] = columns
        if filter_expr is not None:
            kwargs["filter"] = filter_expr
        if implicit_ordering is not None:
            kwargs["implicit_ordering"] = implicit_ordering
        if require_sequenced_output is not None:
            kwargs["require_sequenced_output"] = require_sequenced_output
        options = acero.ScanNodeOptions(dataset, **kwargs)
        decl = acero.Declaration("scan", options)
        return cls(decl, schema=dataset.schema)

    @classmethod
    def table(cls, table: pa.Table) -> Plan:
        """Create a table_source declaration from an in-memory table.

        Parameters
        ----------
        table
            Input Arrow table.

        Returns
        -------
        Plan
            Plan seeded with a table_source declaration.
        """
        decl = acero.Declaration("table_source", acero.TableSourceNodeOptions(table))
        return cls(decl, schema=table.schema)

    @classmethod
    def from_sequence(cls, plans: Sequence[Plan]) -> Plan:
        """Create a plan from a linear sequence of declarations.

        Parameters
        ----------
        plans
            Sequence of plans in order.

        Returns
        -------
        Plan
            Plan wired using Declaration.from_sequence.
        """
        declarations = [plan.declaration for plan in plans]
        decl = acero.Declaration.from_sequence(declarations)
        return cls(decl)

    def project(
        self,
        expressions: Sequence[pc.Expression] | Mapping[str, pc.Expression],
        *,
        names: Sequence[str] | None = None,
    ) -> Plan:
        """Project expressions into a new schema.

        Parameters
        ----------
        expressions
            Expressions or mapping of output names to expressions.
        names
            Optional output names when expressions are positional.

        Returns
        -------
        Plan
            Updated plan with a project node.
        """
        if isinstance(expressions, Mapping):
            expr_list = list(expressions.values())
            names = list(expressions.keys())
        else:
            expr_list = list(expressions)
        options = acero.ProjectNodeOptions(expr_list, names=names)
        decl = acero.Declaration("project", options, inputs=[self.declaration])
        return Plan(decl)

    def filter(self, expr: pc.Expression) -> Plan:
        """Filter rows by an expression.

        Parameters
        ----------
        expr
            Boolean filter expression.

        Returns
        -------
        Plan
            Updated plan with a filter node.
        """
        options = acero.FilterNodeOptions(expr)
        decl = acero.Declaration("filter", options, inputs=[self.declaration])
        return Plan(decl)

    def aggregate(
        self,
        *,
        keys: Sequence[pc.Expression] | None,
        aggregates: Sequence[tuple[object, str, object | None, str]],
    ) -> Plan:
        """Aggregate rows using hash group-by.

        Parameters
        ----------
        keys
            Group-by keys as expressions.
        aggregates
            Aggregation specs of (target, function, options, name).

        Returns
        -------
        Plan
            Updated plan with an aggregate node.
        """
        options = acero.AggregateNodeOptions(aggregates=list(aggregates), keys=keys)
        decl = acero.Declaration("aggregate", options, inputs=[self.declaration])
        return Plan(decl)

    def hash_join(
        self,
        *,
        right: Plan,
        spec: HashJoinSpec,
    ) -> Plan:
        """Join two plans using Acero hashjoin.

        Parameters
        ----------
        right
            Right-hand plan to join.
        spec
            Hash join specification.

        Returns
        -------
        Plan
            Updated plan with a hashjoin node.
        """
        options = acero.HashJoinNodeOptions(
            join_type=spec.how,
            left_keys=list(spec.left_keys),
            right_keys=list(spec.right_keys),
            left_output=list(spec.left_output) if spec.left_output is not None else None,
            right_output=list(spec.right_output) if spec.right_output is not None else None,
            output_suffix_for_left=spec.output_suffix_for_left,
            output_suffix_for_right=spec.output_suffix_for_right,
            filter_expression=spec.filter_expression,
        )
        decl = acero.Declaration(
            "hashjoin",
            options,
            inputs=[self.declaration, right.declaration],
        )
        return Plan(decl)

    def order_by(
        self,
        *,
        sort_keys: Sequence[tuple[str, str]],
        null_placement: str = "at_end",
    ) -> Plan:
        """Apply an order_by node for deterministic ordering.

        Parameters
        ----------
        sort_keys
            Sort keys as (column, order) pairs.
        null_placement
            Null placement policy.

        Returns
        -------
        Plan
            Updated plan with an order_by node.
        """
        options = acero.OrderByNodeOptions(
            sort_keys=list(sort_keys),
            null_placement=null_placement,
        )
        decl = acero.Declaration("order_by", options, inputs=[self.declaration])
        return Plan(decl)

    def to_table(self, *, use_threads: bool = True) -> pa.Table:
        """Materialize the plan as an Arrow table.

        Parameters
        ----------
        use_threads
            Whether to allow compute parallelism.

        Returns
        -------
        pyarrow.Table
            Materialized table result.
        """
        reader = self.declaration.to_reader(use_threads=use_threads)
        return normalize_table_for_compute(reader_to_table(reader))

    def to_reader(self, *, use_threads: bool = True) -> pa.RecordBatchReader:
        """Materialize the plan as an Arrow reader.

        Parameters
        ----------
        use_threads
            Whether to allow compute parallelism.

        Returns
        -------
        pyarrow.RecordBatchReader
            RecordBatchReader for the plan result.
        """
        return self.declaration.to_reader(use_threads=use_threads)


def materialize_plan(plan: Plan, *, use_threads: bool = True) -> pa.Table:
    """Materialize a plan into a normalized Arrow table.

    Parameters
    ----------
    plan
        Plan to materialize.
    use_threads
        Whether to allow compute parallelism.

    Returns
    -------
    pyarrow.Table
        Materialized table with compute-normalized chunks.
    """
    reader = plan.to_reader(use_threads=use_threads)
    return normalize_table_for_compute(reader_to_table(reader))


__all__ = ["HashJoinSpec", "Plan", "materialize_plan"]
