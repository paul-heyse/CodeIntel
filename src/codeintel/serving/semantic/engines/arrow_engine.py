"""Arrow post-processing query engine for semantic serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.streaming import sample_reader
from codeintel.serving.semantic.arrow_plan_builder import ArrowPlanSpec
from codeintel.serving.semantic.duckdb_relation_builder import (
    DuckDBRelationQueryBuilderError,
    RelationBuildContext,
    RelationPlanOptions,
    RelationScanOptions,
    build_relation_plan,
)
from codeintel.serving.semantic.engines.protocol import EngineContext, ExecutablePlan, QueryExplain
from codeintel.serving.semantic.query_ast import ServingQuery
from codeintel.storage.datasets.manifest_index import dataset_schema_for_entry
from codeintel.storage.duckdb_explain import normalize_explain_output

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation

    from codeintel.serving.settings import ServingSettings


class QueryBuilderError(ValueError):
    """Raised when Arrow query construction fails."""


def _fetch_arrow_reader(
    relation: DuckDBPyRelation,
    *,
    batch_size: int,
) -> pa.RecordBatchReader:
    fetcher = getattr(relation, "fetch_arrow_reader", None)
    if callable(fetcher):
        try:
            return fetcher(batch_size)
        except TypeError:
            return fetcher()
    return relation.fetch_record_batch(batch_size)


def _contract_schema_for_table(
    ctx: EngineContext,
    *,
    table_key: str,
) -> pa.Schema | None:
    entry = ctx.dataset_manifests.get(table_key)
    if entry is None:
        return None
    return dataset_schema_for_entry(entry)


def _apply_arrow_plan(
    table: pa.Table,
    *,
    plan_spec: ArrowPlanSpec,
    use_threads: bool,
) -> pa.RecordBatchReader:
    plan = build_table_plan(
        table=table,
        options=TablePlanOptions(
            filter_expr=plan_spec.filter_expr,
            order_by=plan_spec.order_by,
        ),
    )
    if plan_spec.projections:
        plan = plan.project(plan_spec.projections)
    reader = plan.to_reader(use_threads=use_threads)
    if plan_spec.limit is None:
        return reader
    return sample_reader(reader, max_rows=plan_spec.limit)


@dataclass(frozen=True, slots=True)
class ArrowRelationPlan:
    """Executable Arrow post-processing plan wrapper."""

    _relation: DuckDBPyRelation
    _arrow_plan: ArrowPlanSpec
    _use_threads: bool

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Execute the plan and return a RecordBatchReader.

        Returns
        -------
        pyarrow.RecordBatchReader
            Record batch reader for plan results.
        """
        reader = _fetch_arrow_reader(self._relation, batch_size=batch_size)
        table = reader_to_table(reader)
        return _apply_arrow_plan(
            table,
            plan_spec=self._arrow_plan,
            use_threads=self._use_threads,
        )

    def explain(self) -> QueryExplain:
        """Return an EXPLAIN plan for the underlying DuckDB relation.

        Returns
        -------
        QueryExplain
            Query plan summary for the underlying relation.
        """
        plan = normalize_explain_output(self._relation.explain())
        return QueryExplain(sql=self._relation.sql_query(), plan=plan)

    @staticmethod
    def cleanup() -> None:
        """Release temporary resources after execution."""
        return


@dataclass(frozen=True, slots=True)
class ArrowQueryEngine:
    """Arrow post-processing engine backed by DuckDB relations."""

    name: str = "arrow"

    def can_run(self, query: ServingQuery, *, ctx: EngineContext) -> bool:
        """Return True when Arrow post-processing can satisfy the query.

        Returns
        -------
        bool
            True when Arrow post-processing can handle the query.
        """
        if self.name != "arrow":
            return False
        if ctx.warehouse is None:
            return False
        return query.arrow_plan is not None and bool(query.spec.table_key)

    def compile(self, query: ServingQuery, *, ctx: EngineContext) -> ExecutablePlan:
        """Compile a serving query into an Arrow execution plan.

        Returns
        -------
        ExecutablePlan
            Executable Arrow post-processing plan.

        Raises
        ------
        QueryBuilderError
            Raised when the query cannot be compiled for Arrow execution.
        """
        if ctx.warehouse is None:
            msg = f"{self.name} engine requires a warehouse connection"
            raise QueryBuilderError(msg)
        arrow_plan = query.arrow_plan
        if arrow_plan is None:
            msg = "Arrow plan spec is required for Arrow query engine"
            raise QueryBuilderError(msg)
        spec = query.spec
        contract_schema = _contract_schema_for_table(ctx, table_key=spec.table_key)
        try:
            relation = build_relation_plan(
                con=ctx.warehouse.gateway.con,
                spec=spec,
                ast=query.ast,
                context=RelationBuildContext(
                    dataset_manifests=ctx.dataset_manifests,
                    scan_options=_scan_options(ctx.settings),
                    column_types=spec.column_types,
                    contract_schema=contract_schema,
                ),
                options=RelationPlanOptions(
                    plan_spec=query.plan_spec,
                    apply_ast=False,
                ),
            )
            use_threads = _use_threads(ctx.settings)
            return ArrowRelationPlan(
                _relation=relation,
                _arrow_plan=arrow_plan,
                _use_threads=use_threads,
            )
        except DuckDBRelationQueryBuilderError as exc:
            msg = f"Arrow relation plan failed: {exc}"
            raise QueryBuilderError(msg) from exc


def _scan_options(settings: ServingSettings) -> RelationScanOptions:
    return RelationScanOptions(
        batch_size=settings.export_batch_size,
        batch_readahead=settings.dataset_batch_readahead,
        fragment_readahead=settings.dataset_fragment_readahead,
        use_threads=settings.dataset_use_threads,
        unify_schemas=settings.dataset_unify_schemas,
        schema_promote_options=settings.dataset_schema_promote_options,
        metrics_enabled=settings.dataset_scan_metrics_enabled,
    )


def _use_threads(settings: ServingSettings) -> bool:
    resolved = settings.dataset_use_threads
    return True if resolved is None else resolved


__all__ = ["ArrowQueryEngine"]
