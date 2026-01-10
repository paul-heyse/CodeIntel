"""Acero plan helpers for Arrow-first pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Protocol, TypeGuard

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import acero

from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_execution_context,
)
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.normalization import normalize_table_for_compute
from codeintel.core.columnar.ordering import OrderingSpec, SortKey, ordering_keys_present
from codeintel.core.columnar.plan_schema import (
    infer_aggregate_schema,
    infer_filter_schema,
    infer_hash_join_schema,
    infer_order_by_schema,
    infer_project_schema,
)
from codeintel.core.columnar.queryspec import QuerySpec
from codeintel.core.columnar.streaming import configure_arrow_threading_for_context

try:
    from codeintel.core.columnar.external_plans import (
        register_default_external_plan_runners as _register_default_external_plan_runners_impl,
    )
except ImportError:  # pragma: no cover - optional dependency
    _register_default_external_plan_runners_impl = None

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pyarrow.dataset as ds

    from codeintel.core.columnar.arrowdsl import ExecutionPlan
    from codeintel.core.columnar.streaming import DatasetScanOptions

    type TableThunk = Callable[[], pa.Table]
    type ReaderThunk = Callable[[], pa.RecordBatchReader]
else:
    type TableThunk = object
    type ReaderThunk = object

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

    declaration: acero.Declaration | None = None
    table_thunk: TableThunk | None = None
    reader_thunk: ReaderThunk | None = None
    schema: pa.Schema | None = None
    ordering: OrderingSpec | None = None

    def require_declaration(self, *, operation: str) -> acero.Declaration:
        """Return the underlying declaration or raise if missing.

        Parameters
        ----------
        operation
            Operation name for error context.

        Returns
        -------
        pyarrow.acero.Declaration
            Underlying declaration for the plan.

        Raises
        ------
        ValueError
            If the plan has no declaration source.
        """
        if self.declaration is None:
            msg = f"Plan {operation} requires an Acero declaration source."
            raise ValueError(msg)
        return self.declaration

    def _resolved_ordering(self) -> OrderingSpec:
        return self.ordering or OrderingSpec.unordered(reason="unspecified")

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
        if implicit_ordering:
            ordering = OrderingSpec.implicit(reason="scan implicit ordering")
        else:
            ordering = OrderingSpec.unordered(reason="scan unordered")
        return cls(decl, schema=dataset.schema, ordering=ordering)

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
        ordering = OrderingSpec.implicit(reason="table source ordering")
        return cls(decl, schema=table.schema, ordering=ordering)

    @classmethod
    def reader_source(
        cls,
        reader: pa.RecordBatchReader,
        *,
        ordering: OrderingSpec | None = None,
    ) -> Plan:
        """Wrap a record batch reader as a plan source.

        Parameters
        ----------
        reader
            Record batch reader providing plan input.
        ordering
            Optional ordering metadata for the reader source.

        Returns
        -------
        Plan
            Plan backed by a reader thunk.
        """
        resolved_ordering = ordering or OrderingSpec.implicit(reason="reader source ordering")
        return cls(
            reader_thunk=lambda: reader,
            schema=reader.schema,
            ordering=resolved_ordering,
        )

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
        declarations = [
            plan.require_declaration(operation="from_sequence") for plan in plans
        ]
        decl = acero.Declaration.from_sequence(declarations)
        ordering = plans[-1].ordering if plans else None
        schema = plans[-1].schema if plans else None
        return cls(decl, schema=schema, ordering=ordering)

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
        decl = acero.Declaration(
            "project",
            options,
            inputs=[self.require_declaration(operation="project")],
        )
        ordering = _project_ordering(
            self._resolved_ordering(),
            expressions=expr_list,
            names=names,
        )
        schema = infer_project_schema(self.schema, expr_list, names=names)
        return Plan(decl, schema=schema, ordering=ordering)

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
        decl = acero.Declaration(
            "filter",
            options,
            inputs=[self.require_declaration(operation="filter")],
        )
        ordering = _filter_ordering(self._resolved_ordering())
        schema = infer_filter_schema(self.schema)
        return Plan(decl, schema=schema, ordering=ordering)

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
        decl = acero.Declaration(
            "aggregate",
            options,
            inputs=[self.require_declaration(operation="aggregate")],
        )
        ordering = _aggregate_ordering(self._resolved_ordering(), keys=keys)
        schema = infer_aggregate_schema(self.schema, keys=keys, aggregates=aggregates)
        return Plan(decl, schema=schema, ordering=ordering)

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
            inputs=[
                self.require_declaration(operation="hash_join (left)"),
                right.require_declaration(operation="hash_join (right)"),
            ],
        )
        ordering = _merge_join_ordering(
            left=self._resolved_ordering(),
            right=right._resolved_ordering(),
            spec=spec,
            left_columns=_resolve_join_columns(self, output=spec.left_output),
            right_columns=_resolve_join_columns(right, output=spec.right_output),
        )
        schema = infer_hash_join_schema(self.schema, right.schema, spec=spec)
        return Plan(decl, schema=schema, ordering=ordering)

    def order_by(
        self,
        *,
        sort_keys: Sequence[SortKey],
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
        decl = acero.Declaration(
            "order_by",
            options,
            inputs=[self.require_declaration(operation="order_by")],
        )
        ordering = OrderingSpec.explicit(
            keys=sort_keys,
            reason="order_by explicit ordering",
            pipeline_breaker=True,
        )
        schema = infer_order_by_schema(self.schema)
        return Plan(decl, schema=schema, ordering=ordering)

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

        Raises
        ------
        ValueError
            If the plan has no declaration or source thunk.

        Notes
        -----
        Deprecated. Prefer ``ExecutionPlan.from_plan(plan)`` with an
        ``ExecutionContext`` to preserve ordering metadata.
        """
        configure_arrow_threading_for_context(ctx=None)
        if self.declaration is not None:
            reader = self.declaration.to_reader(use_threads=use_threads)
            return normalize_table_for_compute(reader_to_table(reader))
        if self.table_thunk is not None:
            return self.table_thunk()
        if self.reader_thunk is not None:
            return normalize_table_for_compute(reader_to_table(self.reader_thunk()))
        msg = "Plan has no declaration, table thunk, or reader thunk."
        raise ValueError(msg)

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

        Raises
        ------
        ValueError
            If the plan has no declaration or source thunk.

        Notes
        -----
        Deprecated. Prefer ``ExecutionPlan.from_plan(plan)`` with an
        ``ExecutionContext`` to preserve ordering metadata.
        """
        configure_arrow_threading_for_context(ctx=None)
        if self.declaration is not None:
            return self.declaration.to_reader(use_threads=use_threads)
        if self.reader_thunk is not None:
            return self.reader_thunk()
        if self.table_thunk is not None:
            return self.table_thunk().to_reader()
        msg = "Plan has no declaration, table thunk, or reader thunk."
        raise ValueError(msg)


def _field_name_for_expression(expr: pc.Expression) -> str | None:
    candidate = str(expr)
    if expr.equals(E.field(candidate)):
        return candidate
    return None


def _project_output_names(
    expressions: Sequence[pc.Expression],
    *,
    names: Sequence[str] | None,
) -> list[str | None]:
    if names is not None:
        return list(names)
    return [_field_name_for_expression(expr) for expr in expressions]


def _project_field_mapping(
    expressions: Sequence[pc.Expression],
    *,
    names: Sequence[str] | None,
) -> dict[str, str]:
    output_names = _project_output_names(expressions, names=names)
    mapping: dict[str, str] = {}
    for expr, output_name in zip(expressions, output_names, strict=True):
        if output_name is None:
            continue
        field_name = _field_name_for_expression(expr)
        if field_name is None:
            continue
        mapping[field_name] = output_name
    return mapping


def _project_ordering(
    ordering: OrderingSpec,
    *,
    expressions: Sequence[pc.Expression],
    names: Sequence[str] | None,
) -> OrderingSpec:
    if ordering.level == "unordered" or not ordering.keys:
        return ordering
    mapping = _project_field_mapping(expressions, names=names)
    if not mapping:
        return OrderingSpec.unordered(reason="project drops ordering")
    new_keys: list[SortKey] = []
    for name, direction in ordering.keys:
        output_name = mapping.get(name)
        if output_name is None:
            return OrderingSpec.unordered(reason="project drops ordering")
        new_keys.append((output_name, direction))
    return replace(ordering, keys=tuple(new_keys))


def _filter_ordering(ordering: OrderingSpec) -> OrderingSpec:
    if ordering.level == "unordered":
        return ordering
    if ordering.reason is not None:
        return ordering
    return replace(ordering, reason="filter preserves ordering")


def _aggregate_ordering(
    ordering: OrderingSpec,
    *,
    keys: Sequence[pc.Expression] | None,
) -> OrderingSpec:
    reason = "aggregate pipeline breaker"
    if ordering.level != "unordered":
        reason = "aggregate pipeline breaker (drops ordering)"
    if not keys:
        reason = f"{reason} (no keys)"
    return OrderingSpec.unordered(
        reason=reason,
        pipeline_breaker=True,
    )


def _resolve_join_columns(plan: Plan, *, output: Sequence[str] | None) -> Sequence[str] | None:
    if output is not None:
        return tuple(output)
    schema = plan.schema
    if schema is None:
        return None
    return tuple(schema.names)


def _merge_join_ordering(
    *,
    left: OrderingSpec,
    right: OrderingSpec,
    spec: HashJoinSpec,
    left_columns: Sequence[str] | None,
    right_columns: Sequence[str] | None,
) -> OrderingSpec:
    left_join_types = {"left outer", "left semi", "left anti", "inner"}
    right_join_types = {"right outer", "right semi", "right anti"}
    left_ordered = ordering_keys_present(left, left_columns)
    right_ordered = ordering_keys_present(right, right_columns)
    if spec.how in left_join_types and left_ordered:
        return replace(
            left,
            pipeline_breaker=True,
            reason="hash join preserves left ordering",
        )
    if spec.how in right_join_types and right_ordered:
        return replace(
            right,
            pipeline_breaker=True,
            reason="hash join preserves right ordering",
        )
    return OrderingSpec.unordered(
        reason="hash join output",
        pipeline_breaker=True,
    )


def materialize_plan(
    plan: Plan,
    *,
    ctx: ExecutionContext | None = None,
    use_threads: bool = True,
    combine_chunks: bool = True,
) -> pa.Table:
    """Materialize a plan into a normalized Arrow table.

    Parameters
    ----------
    plan
        Plan to materialize.
    ctx
        Optional execution context overrides.
    use_threads
        Whether to allow compute parallelism.
    combine_chunks
        Whether to combine chunks after materialization.

    Returns
    -------
    pyarrow.Table
        Materialized table with compute-normalized chunks.

    Raises
    ------
    ValueError
        If the plan has no declaration or source thunk.

    Notes
    -----
    Deprecated. Prefer ``ExecutionPlan.from_plan(plan)`` with an
    ``ExecutionContext`` to preserve ordering metadata.
    """
    execution_ctx = resolve_execution_context(ctx)
    if ctx is None:
        execution_ctx = replace(
            execution_ctx,
            use_threads=use_threads,
            combine_chunks=combine_chunks,
        )
    configure_arrow_threading_for_context(ctx=execution_ctx)
    if plan.declaration is not None:
        reader = plan.declaration.to_reader(use_threads=execution_ctx.resolve_use_threads())
    elif plan.reader_thunk is not None:
        reader = plan.reader_thunk()
    elif plan.table_thunk is not None:
        reader = plan.table_thunk().to_reader()
    else:
        msg = "Plan has no declaration, table thunk, or reader thunk."
        raise ValueError(msg)
    table = reader_to_table(reader)
    return normalize_table_for_compute(table, combine_chunks=execution_ctx.combine_chunks)


@dataclass(frozen=True, slots=True)
class ScanPlanOptions:
    """Options for building scan plans."""

    columns: Sequence[str] | Mapping[str, pc.Expression] | None = None
    filter_expr: pc.Expression | None = None
    implicit_ordering: bool | None = None
    require_sequenced_output: bool | None = None
    order_by: Sequence[SortKey] | None = None


@dataclass(frozen=True, slots=True)
class QueryPlanOptions:
    """Options for building query plans."""

    provenance: bool = False
    implicit_ordering: bool | None = None
    require_sequenced_output: bool | None = None
    order_by: Sequence[SortKey] | None = None


def query_plan_options_for_context(
    *,
    ctx: ExecutionContext | None,
    options: QueryPlanOptions | None = None,
) -> QueryPlanOptions:
    """Return query plan options with provenance derived from an execution context.

    Returns
    -------
    QueryPlanOptions
        Query plan options with provenance updated from context.
    """
    resolved = options or QueryPlanOptions()
    resolved_ctx = resolve_execution_context(ctx)
    provenance = resolved.provenance or resolved_ctx.provenance
    implicit_ordering = resolved.implicit_ordering
    require_sequenced_output = resolved.require_sequenced_output
    determinism = resolved_ctx.resolve_determinism()
    profile = resolved_ctx.runtime_profile
    if profile is not None:
        provenance = profile.resolve_provenance(default=provenance)
        implicit_ordering = profile.resolve_implicit_ordering(default=implicit_ordering)
        require_sequenced_output = profile.resolve_require_sequenced_output(
            default=require_sequenced_output
        )
    if determinism == "canonical":
        provenance = True
        if implicit_ordering is None:
            implicit_ordering = True
        if require_sequenced_output is None:
            require_sequenced_output = True
    else:
        implicit_ordering = None
        require_sequenced_output = None
    if (
        provenance == resolved.provenance
        and implicit_ordering == resolved.implicit_ordering
        and require_sequenced_output == resolved.require_sequenced_output
    ):
        return resolved
    return replace(
        resolved,
        provenance=provenance,
        implicit_ordering=implicit_ordering,
        require_sequenced_output=require_sequenced_output,
    )


def build_scan_plan(
    dataset: ds.Dataset,
    *,
    options: ScanPlanOptions | None = None,
) -> Plan:
    """Return a scan plan with explicit project/filter nodes.

    Parameters
    ----------
    dataset
        Dataset to scan.
    options
        Scan plan options for projection, filtering, and ordering.

    Returns
    -------
    Plan
        Fused scan/project/filter plan for the dataset.
    """
    resolved = options or ScanPlanOptions()
    plan = Plan.scan(
        dataset,
        columns=resolved.columns,
        filter_expr=resolved.filter_expr,
        implicit_ordering=resolved.implicit_ordering,
        require_sequenced_output=resolved.require_sequenced_output,
    )
    projection = _projection_for_columns(resolved.columns)
    if projection is not None:
        expressions, names = projection
        plan = plan.project(expressions, names=names)
    if resolved.filter_expr is not None:
        plan = plan.filter(resolved.filter_expr)
    if resolved.order_by is not None:
        plan = plan.order_by(sort_keys=resolved.order_by)
    return plan


def build_query_plan(
    dataset: ds.Dataset,
    *,
    spec: QuerySpec,
    options: QueryPlanOptions | None = None,
) -> Plan:
    """Return a scan plan compiled from a QuerySpec.

    Parameters
    ----------
    dataset
        Dataset to scan.
    spec
        Query specification for predicates and projection.
    options
        Query plan options for provenance, ordering, and scan behavior.

    Returns
    -------
    Plan
        Compiled scan/filter/project plan.
    """
    resolved = options or QueryPlanOptions()
    scan_filter = spec.scan_filter_expression()
    plan = Plan.scan(
        dataset,
        columns=spec.scan_columns(provenance=resolved.provenance),
        filter_expr=scan_filter,
        implicit_ordering=resolved.implicit_ordering,
        require_sequenced_output=resolved.require_sequenced_output,
    )
    post_filter = spec.post_filter_expression()
    if post_filter is not None:
        plan = plan.filter(post_filter)
    projection = spec.project_expressions(provenance=resolved.provenance)
    if projection:
        plan = plan.project(projection)
    if resolved.order_by is not None:
        plan = plan.order_by(sort_keys=resolved.order_by)
    return plan


def build_query_plan_for_context(
    dataset: ds.Dataset,
    *,
    spec: QuerySpec,
    ctx: ExecutionContext | None = None,
    options: QueryPlanOptions | None = None,
) -> Plan:
    """Return a scan plan compiled from a QuerySpec and execution context.

    Returns
    -------
    Plan
        Compiled scan/filter/project plan.
    """
    resolved = query_plan_options_for_context(ctx=ctx, options=options)
    return build_query_plan(dataset, spec=spec, options=resolved)


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


@dataclass(frozen=True, slots=True)
class ExternalPlanRequest:
    """Execution request for external plan runners."""

    spec: ExternalPlanSpec
    dataset: ds.Dataset | None
    filter_expr: ds.Expression | None
    columns: Sequence[str] | Mapping[str, pc.Expression] | None
    schema: pa.Schema | None
    scan_options: DatasetScanOptions | None
    use_threads: bool | None


class ExternalPlanRunner(Protocol):
    """Protocol for external plan runners."""

    def __call__(
        self,
        *,
        request: ExternalPlanRequest,
    ) -> pa.RecordBatchReader | ReaderThunk | ExecutionPlan:
        """Execute an external plan and return a reader or execution plan.

        Parameters
        ----------
        request
            External plan execution request.

        Returns
        -------
        pyarrow.RecordBatchReader | ReaderThunk | ExecutionPlan
            Record batch reader, thunk returning the reader, or an execution plan.
        """
        ...


_EXTERNAL_PLAN_RUNNERS: dict[str, ExternalPlanRunner] = {}


class _ExecutionPlanLike(Protocol):
    external_request: ExternalPlanRequest | None
    reader_thunk: object | None
    table_thunk: object | None

    def to_reader(self, *, ctx: ExecutionContext) -> pa.RecordBatchReader:
        """Return a reader for the execution plan."""


def _is_execution_plan_like(value: object) -> TypeGuard[_ExecutionPlanLike]:
    return (
        hasattr(value, "to_reader")
        and hasattr(value, "external_request")
        and hasattr(value, "reader_thunk")
        and hasattr(value, "table_thunk")
    )


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


def run_external_plan(request: ExternalPlanRequest) -> pa.RecordBatchReader:
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
    normalized = _normalize_external_engine(request.spec.engine)
    runner = _EXTERNAL_PLAN_RUNNERS.get(normalized)
    if runner is None:
        _register_default_external_plan_runners()
        runner = _EXTERNAL_PLAN_RUNNERS.get(normalized)
    if runner is None:
        msg = f"No external plan runner registered for engine '{normalized}'."
        raise ValueError(msg)
    result = runner(request=request)
    if isinstance(result, pa.RecordBatchReader):
        return result
    if _is_execution_plan_like(result):
        execution_ctx = resolve_execution_context(None)
        if request.use_threads is not None:
            execution_ctx = replace(execution_ctx, use_threads=request.use_threads)
        return result.to_reader(ctx=execution_ctx)
    if callable(result):
        reader = result()
        if isinstance(reader, pa.RecordBatchReader):
            return reader
        msg = "External plan reader thunk did not return a RecordBatchReader."
        raise TypeError(msg)
    msg = "External plan runner did not return a RecordBatchReader."
    raise TypeError(msg)


def _register_default_external_plan_runners() -> None:
    if _register_default_external_plan_runners_impl is None:
        return
    _register_default_external_plan_runners_impl()


__all__ = [
    "ExternalPlanRequest",
    "ExternalPlanRunner",
    "ExternalPlanSpec",
    "HashJoinSpec",
    "Plan",
    "QueryPlanOptions",
    "ScanPlanOptions",
    "build_query_plan",
    "build_query_plan_for_context",
    "build_scan_plan",
    "list_external_plan_runners",
    "materialize_plan",
    "query_plan_options_for_context",
    "register_external_plan_runner",
    "run_external_plan",
]
