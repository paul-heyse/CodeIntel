"""Acero plan helpers for Arrow-first pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import acero

from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.normalization import normalize_table_for_compute

if TYPE_CHECKING:
    import pyarrow.dataset as ds

    from codeintel.core.columnar.streaming import DatasetScanOptions

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


def materialize_plan(
    plan: Plan,
    *,
    use_threads: bool = True,
    combine_chunks: bool = True,
) -> pa.Table:
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
    return normalize_table_for_compute(reader_to_table(reader), combine_chunks=combine_chunks)


def build_scan_plan(
    dataset: ds.Dataset,
    *,
    columns: Sequence[str] | Mapping[str, pc.Expression] | None,
    filter_expr: pc.Expression | None,
    implicit_ordering: bool | None = None,
    require_sequenced_output: bool | None = None,
    order_by: Sequence[tuple[str, str]] | None = None,
) -> Plan:
    """Return a scan plan with explicit project/filter nodes.

    Parameters
    ----------
    dataset
        Dataset to scan.
    columns
        Projection columns or expressions for pushdown and project nodes.
    filter_expr
        Filter expression to apply via scan pushdown and filter node.
    implicit_ordering
        Whether to request implicit ordering from the scan node.
    require_sequenced_output
        Whether to require sequenced output batches from the scan node.
    order_by
        Optional order-by keys to apply in the plan.

    Returns
    -------
    Plan
        Fused scan/project/filter plan for the dataset.
    """
    plan = Plan.scan(
        dataset,
        columns=columns,
        filter_expr=filter_expr,
        implicit_ordering=implicit_ordering,
        require_sequenced_output=require_sequenced_output,
    )
    projection = _projection_for_columns(columns)
    if projection is not None:
        expressions, names = projection
        plan = plan.project(expressions, names=names)
    if filter_expr is not None:
        plan = plan.filter(filter_expr)
    if order_by is not None:
        plan = plan.order_by(sort_keys=order_by)
    return plan


def _projection_for_columns(
    columns: Sequence[str] | Mapping[str, pc.Expression] | None,
) -> tuple[Sequence[pc.Expression] | Mapping[str, pc.Expression], Sequence[str] | None] | None:
    if columns is None:
        return None
    if isinstance(columns, Mapping):
        return dict(columns), None
    names = list(columns)
    expressions = [E.field(name) for name in names]
    return expressions, names


@dataclass(frozen=True, slots=True)
class ExternalPlanSpec:
    """External plan specification for non-Acero execution."""

    engine: str
    payload: object
    metadata: Mapping[str, object] = field(default_factory=dict)


class ExternalPlanRunner(Protocol):
    """Protocol for external plan runners."""

    def __call__(
        self,
        *,
        spec: ExternalPlanSpec,
        dataset: ds.Dataset | None,
        filter_expr: ds.Expression | None,
        columns: Sequence[str] | Mapping[str, pc.Expression] | None,
        scan_options: DatasetScanOptions | None,
        use_threads: bool | None,
    ) -> pa.RecordBatchReader:
        """Execute an external plan and return a record batch reader.

        Parameters
        ----------
        spec
            External plan specification.
        dataset
            Dataset for plan execution.
        filter_expr
            Optional dataset filter expression.
        columns
            Columns or expression mapping to project.
        scan_options
            Dataset scan options for execution.
        use_threads
            Whether to enable threaded execution.

        Returns
        -------
        pyarrow.RecordBatchReader
            Record batch reader for plan results.
        """
        ...


_EXTERNAL_PLAN_RUNNERS: dict[str, ExternalPlanRunner] = {}


def _normalize_external_engine(engine: str) -> str:
    normalized = engine.strip().lower()
    if not normalized:
        msg = "External plan engine name must be non-empty."
        raise ValueError(msg)
    return normalized


def register_external_plan_runner(name: str, runner: ExternalPlanRunner) -> None:
    """Register an external plan runner under a normalized engine name."""
    normalized = _normalize_external_engine(name)
    _EXTERNAL_PLAN_RUNNERS[normalized] = runner


def list_external_plan_runners() -> tuple[str, ...]:
    """Return the registered external plan runner names.

    Returns
    -------
    tuple[str, ...]
        Sorted external runner names.
    """
    return tuple(sorted(_EXTERNAL_PLAN_RUNNERS))


def run_external_plan(
    spec: ExternalPlanSpec,
    *,
    dataset: ds.Dataset | None,
    filter_expr: ds.Expression | None,
    columns: Sequence[str] | Mapping[str, pc.Expression] | None,
    scan_options: DatasetScanOptions | None,
    use_threads: bool | None,
) -> pa.RecordBatchReader:
    """Execute an external plan via the registered runner.

    Returns
    -------
    pyarrow.RecordBatchReader
        Record batch reader for plan results.

    Raises
    ------
    ValueError
        Raised when no runner is registered for the plan engine.
    TypeError
        Raised when the runner returns an unexpected type.
    """
    normalized = _normalize_external_engine(spec.engine)
    runner = _EXTERNAL_PLAN_RUNNERS.get(normalized)
    if runner is None:
        msg = f"No external plan runner registered for engine '{normalized}'."
        raise ValueError(msg)
    reader = runner(
        spec=spec,
        dataset=dataset,
        filter_expr=filter_expr,
        columns=columns,
        scan_options=scan_options,
        use_threads=use_threads,
    )
    if isinstance(reader, pa.RecordBatchReader):
        return reader
    if isinstance(reader, pa.Table):
        to_reader = getattr(reader, "to_reader", None)
        if callable(to_reader):
            return to_reader()
    msg = "External plan runner did not return a RecordBatchReader."
    raise TypeError(msg)


__all__ = [
    "ExternalPlanRunner",
    "ExternalPlanSpec",
    "HashJoinSpec",
    "Plan",
    "build_scan_plan",
    "list_external_plan_runners",
    "materialize_plan",
    "register_external_plan_runner",
    "run_external_plan",
]
