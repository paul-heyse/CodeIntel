"""DuckDB relation-first semantic query engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.serving.semantic.duckdb_relation_builder import (
    DuckDBRelationQueryBuilderError,
    RelationBuildContext,
    RelationScanOptions,
    build_relation_plan,
)
from codeintel.serving.semantic.engines.polars_engine import (
    PolarsPlanAdapter,
    PolarsQueryBuilderError,
)
from codeintel.serving.semantic.engines.protocol import EngineContext, ExecutablePlan, QueryExplain
from codeintel.serving.semantic.query_ast import ServingQuery
from codeintel.storage.datasets.manifest_index import dataset_schema_for_entry
from codeintel.storage.duckdb_explain import normalize_explain_output
from codeintel.storage.protocols.duckdb_relation import adapt_duckdb_relation_stream

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection, DuckDBPyRelation

    from codeintel.serving.settings import ServingSettings
    from codeintel.storage.protocols.export import ResultStream

LOG = logging.getLogger(__name__)

_RESULT_ENGINE_POLARS = "polars"


def cleanup_temp_tables_if_needed(*, con: DuckDBPyConnection, temp_tables: Sequence[str]) -> None:
    """Unregister temp tables when supported by the backend."""
    unregister = getattr(con, "unregister", None)
    if not callable(unregister):
        return
    for table_name in temp_tables:
        unregister(table_name)


def _fetch_arrow_reader(
    relation_or_con: DuckDBPyRelation | DuckDBPyConnection,
    *,
    batch_size: int,
) -> pa.RecordBatchReader:
    fetcher = getattr(relation_or_con, "fetch_arrow_reader", None)
    if callable(fetcher):
        try:
            return fetcher(batch_size)
        except TypeError:
            return fetcher()
    return relation_or_con.fetch_record_batch(batch_size)


def _contract_schema_for_table(
    ctx: EngineContext,
    *,
    table_key: str,
) -> pa.Schema | None:
    entry = ctx.dataset_manifests.get(table_key)
    if entry is None:
        return None
    return dataset_schema_for_entry(entry)


def _polars_reader(
    *,
    relation: DuckDBPyRelation,
    settings: ServingSettings,
    batch_size: int,
) -> pa.RecordBatchReader:
    adapter = PolarsPlanAdapter(settings=settings)
    plan = adapter.build(relation=relation)
    try:
        return plan.to_reader(batch_size=batch_size)
    except PolarsQueryBuilderError as exc:
        LOG.warning("Polars result engine failed; falling back to DuckDB: %s", exc)
        return _fetch_arrow_reader(relation, batch_size=batch_size)


class QueryBuilderError(ValueError):
    """Raised when query construction fails."""


@dataclass(frozen=True, slots=True)
class DuckDBRelationPlan:
    """Executable DuckDB relation plan wrapper."""

    _relation: DuckDBPyRelation
    _stream: ResultStream
    settings: ServingSettings

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Execute the plan and return a RecordBatchReader.

        Returns
        -------
        pyarrow.RecordBatchReader
            Reader over the plan output.
        """
        if self.settings.result_engine.lower() == _RESULT_ENGINE_POLARS:
            return _polars_reader(
                relation=self._relation,
                settings=self.settings,
                batch_size=batch_size,
            )
        return self._stream.to_reader(batch_size=batch_size)

    def explain(self) -> QueryExplain:
        """Return an EXPLAIN plan for the relation.

        Returns
        -------
        QueryExplain
            Explain payload with SQL and plan text.
        """
        plan = normalize_explain_output(self._relation.explain())
        return QueryExplain(sql=self._relation.sql_query(), plan=plan)

    @staticmethod
    def cleanup() -> None:
        """Release temporary resources after execution."""
        return


@dataclass(frozen=True, slots=True)
class DuckDBQueryEngine:
    """DuckDB engine backed by relations."""

    name: str = "duckdb"

    def can_run(self, query: ServingQuery, *, ctx: EngineContext) -> bool:
        """Return True when DuckDB can satisfy the query.

        Parameters
        ----------
        query
            Serving query bundle with AST/spec data.
        ctx
            Engine context with warehouse access.

        Returns
        -------
        bool
            True if the engine can execute the query.
        """
        spec = query.spec
        return ctx.warehouse is not None and self.name.lower() == "duckdb" and bool(spec.table_key)

    def compile(self, query: ServingQuery, *, ctx: EngineContext) -> ExecutablePlan:
        """Compile a serving query into a DuckDB execution plan.

        Parameters
        ----------
        query
            Serving query bundle with AST/spec data.
        ctx
            Engine context with warehouse access.

        Returns
        -------
        ExecutablePlan
            Executable DuckDB plan wrapper.

        Raises
        ------
        QueryBuilderError
            If the warehouse is unavailable or the relation plan fails.
        """
        if ctx.warehouse is None:
            msg = f"{self.name} engine requires a warehouse connection"
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
                    scan_options=RelationScanOptions(
                        batch_size=ctx.settings.export_batch_size,
                        batch_readahead=ctx.settings.dataset_batch_readahead,
                        fragment_readahead=ctx.settings.dataset_fragment_readahead,
                        use_threads=ctx.settings.dataset_use_threads,
                        unify_schemas=ctx.settings.dataset_unify_schemas,
                        schema_promote_options=ctx.settings.dataset_schema_promote_options,
                        metrics_enabled=ctx.settings.dataset_scan_metrics_enabled,
                    ),
                    column_types=spec.column_types,
                    contract_schema=contract_schema,
                ),
                plan_spec=query.plan_spec,
            )
            return DuckDBRelationPlan(
                _relation=relation,
                _stream=adapt_duckdb_relation_stream(relation),
                settings=ctx.settings,
            )
        except DuckDBRelationQueryBuilderError as exc:
            msg = f"DuckDB relation plan failed: {exc}"
            raise QueryBuilderError(msg) from exc


__all__ = [
    "DuckDBQueryEngine",
    "cleanup_temp_tables_if_needed",
]
