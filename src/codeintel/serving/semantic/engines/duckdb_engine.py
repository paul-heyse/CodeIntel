"""DuckDB relation-first semantic query engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
from sqlglot.errors import SqlglotError

from codeintel.core.columnar.schema import unify_schema_for_batches
from codeintel.serving.semantic.duckdb_relation_builder import (
    DuckDBRelationQueryBuilderError,
    RelationScanOptions,
    build_relation_plan,
)
from codeintel.serving.semantic.engines.protocol import EngineContext, ExecutablePlan, QueryExplain
from codeintel.serving.semantic.guardrails import warn_eager_materialization
from codeintel.serving.semantic.query_ast import ServingQuery
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.duckdb_types import DuckDBError
from codeintel.storage.queries.safe import SqlIngressPolicy, UnsafeSqlError, assert_select_perimeter
from codeintel.storage.schema import arrow_schema_for_table_key
from codeintel.storage.sqlglot_tools import render_sql_duckdb

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection, DuckDBPyRelation

    from codeintel.serving.semantic.specs import SemanticQuerySpec
    from codeintel.storage.warehouse import Warehouse


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
    if ctx.warehouse is None:
        return None
    try:
        return arrow_schema_for_table_key(
            ctx.warehouse.gateway.con,
            table_key=table_key,
            repo=ctx.pointer.repo,
            commit=ctx.pointer.commit,
        )
    except (DuckDBError, RuntimeError, TypeError, ValueError):
        return None


class QueryBuilderError(ValueError):
    """Raised when query construction fails."""


@dataclass(frozen=True, slots=True)
class DuckDBRelationPlan:
    """Executable DuckDB relation plan wrapper."""

    relation: DuckDBPyRelation
    warehouse: Warehouse

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Execute the plan and return a RecordBatchReader.

        Returns
        -------
        pyarrow.RecordBatchReader
            Reader over the plan output.
        """
        return _fetch_arrow_reader(self.relation, batch_size=batch_size)

    def to_table(self) -> pa.Table:
        """Execute the plan and return an Arrow table.

        Returns
        -------
        pyarrow.Table
            Materialized Arrow table.
        """
        warn_eager_materialization(engine="duckdb", context="duckdb_relation_plan")
        reader = _fetch_arrow_reader(self.relation, batch_size=DEFAULT_ARROW_BATCH_SIZE)
        batches = list(reader)
        schema = unify_schema_for_batches(batches, base_schema=reader.schema)
        return pa.Table.from_batches(batches, schema=schema)

    def explain(self) -> QueryExplain:
        """Return an EXPLAIN plan for the relation.

        Returns
        -------
        QueryExplain
            Explain payload with SQL and plan text.
        """
        return QueryExplain(sql=self.relation.sql_query(), plan=self.relation.explain())

    @staticmethod
    def cleanup() -> None:
        """Release temporary resources after execution."""
        return


@dataclass(frozen=True, slots=True)
class DuckDBSqlPlan:
    """Executable DuckDB SQL plan wrapper."""

    sql: str
    temp_tables: tuple[str, ...]
    warehouse: Warehouse

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Execute the plan and return a RecordBatchReader.

        Returns
        -------
        pyarrow.RecordBatchReader
            Reader over the plan output.
        """
        result = self._execute()
        return _fetch_arrow_reader(result, batch_size=batch_size)

    def to_table(self) -> pa.Table:
        """Execute the plan and return an Arrow table.

        Returns
        -------
        pyarrow.Table
            Materialized Arrow table.
        """
        warn_eager_materialization(engine="duckdb", context="duckdb_sql_plan")
        result = self._execute()
        reader = _fetch_arrow_reader(result, batch_size=DEFAULT_ARROW_BATCH_SIZE)
        batches = list(reader)
        schema = unify_schema_for_batches(batches, base_schema=reader.schema)
        return pa.Table.from_batches(batches, schema=schema)

    def explain(self) -> QueryExplain:
        """Return an EXPLAIN plan for the SQL.

        Returns
        -------
        QueryExplain
            Explain payload with SQL and plan text.
        """
        relation = self.warehouse.gateway.con.sql(self.sql)
        plan_text = relation.explain()
        return QueryExplain(sql=self.sql, plan=plan_text)

    def cleanup(self) -> None:
        """Release temporary resources after execution."""
        cleanup_temp_tables_if_needed(
            con=self.warehouse.gateway.con,
            temp_tables=self.temp_tables,
        )

    def _execute(self) -> DuckDBPyConnection:
        return self.warehouse.gateway.policy.execute_sql(self.sql)


@dataclass(frozen=True, slots=True)
class DuckDBQueryEngine:
    """DuckDB engine backed by relations with SQLGlot fallback."""

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
            If the warehouse is unavailable or SQL is unsafe.
        """
        if ctx.warehouse is None:
            msg = f"{self.name} engine requires a warehouse connection"
            raise QueryBuilderError(msg)
        spec = query.spec
        relation_error: DuckDBRelationQueryBuilderError | None = None
        contract_schema = _contract_schema_for_table(ctx, table_key=spec.table_key)
        try:
            relation = build_relation_plan(
                con=ctx.warehouse.gateway.con,
                spec=spec,
                dataset_manifests=ctx.dataset_manifests,
                scan_options=RelationScanOptions(
                    batch_size=ctx.settings.export_batch_size,
                    fragment_readahead=ctx.settings.dataset_fragment_readahead,
                    metrics_enabled=ctx.settings.dataset_scan_metrics_enabled,
                ),
                column_types=spec.column_types,
                contract_schema=contract_schema,
            )
            return DuckDBRelationPlan(relation=relation, warehouse=ctx.warehouse)
        except DuckDBRelationQueryBuilderError as exc:
            relation_error = exc

        sqlglot_error: Exception | None = None
        try:
            sql = render_sql_duckdb(query.ast)
            policy = _sql_policy_for_spec(spec, temp_tables=())
            validated = _validate_sql(sql, policy=policy)
        except UnsafeSqlError as exc:
            raise QueryBuilderError(str(exc)) from exc
        except (SqlglotError, TypeError, ValueError) as exc:
            sqlglot_error = exc
        else:
            return DuckDBSqlPlan(
                sql=validated,
                temp_tables=(),
                warehouse=ctx.warehouse,
            )

        if relation_error is None:
            msg = "DuckDB relation plan failed with an unknown error"
            raise QueryBuilderError(msg)
        msg = f"DuckDB relation plan failed: {relation_error}; SQLGlot error: {sqlglot_error}"
        if sqlglot_error is not None:
            raise QueryBuilderError(msg) from sqlglot_error
        raise QueryBuilderError(msg) from relation_error


def _sql_policy_for_spec(
    spec: SemanticQuerySpec, *, temp_tables: tuple[str, ...]
) -> SqlIngressPolicy:
    allowed_tables = {spec.table_key.lower(), *{name.lower() for name in temp_tables}}
    return SqlIngressPolicy(
        allowed_tables=frozenset(allowed_tables),
        allowed_functions=_SEMANTIC_SQL_ALLOWED_FUNCTIONS,
    )


def _validate_sql(sql: str, *, policy: SqlIngressPolicy) -> str:
    root = assert_select_perimeter(sql, policy=policy)
    return render_sql_duckdb(root)


_SEMANTIC_SQL_ALLOWED_FUNCTIONS = frozenset(
    {
        "abs",
        "cast",
        "coalesce",
        "contains",
        "date_add",
        "date_diff",
        "date_sub",
        "date_trunc",
        "floor",
        "length",
        "lower",
        "ltrim",
        "nullif",
        "round",
        "rtrim",
        "starts_with",
        "strftime",
        "substr",
        "substring",
        "trim",
        "upper",
    }
)


__all__ = [
    "DuckDBQueryEngine",
    "DuckDBRelationPlan",
    "DuckDBSqlPlan",
    "cleanup_temp_tables_if_needed",
]
