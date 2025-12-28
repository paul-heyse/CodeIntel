"""DuckDB relation-first semantic query engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
from ibis.common.exceptions import IbisError
from sqlglot.errors import SqlglotError

from codeintel.serving.semantic.duckdb_relation_builder import (
    DuckDBRelationQueryBuilderError,
    build_relation_plan,
)
from codeintel.serving.semantic.engines.protocol import EngineContext, ExecutablePlan, QueryExplain
from codeintel.serving.semantic.query_builder import QueryBuilderError, build_query
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.semantic.sqlglot_query_builder import (
    SqlglotQueryBuilderError,
    build_sqlglot_query,
)
from codeintel.storage.gateway import ibis_facade
from codeintel.storage.queries.safe import SqlIngressPolicy, UnsafeSqlError, assert_select_perimeter
from codeintel.storage.sqlglot_tools import render_sql_duckdb

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection, DuckDBPyRelation

    from codeintel.storage.warehouse import Warehouse


def cleanup_temp_tables_if_needed(*, con: DuckDBPyConnection, temp_tables: Sequence[str]) -> None:
    """Unregister Ibis temp tables when supported by the backend."""
    unregister = getattr(con, "unregister", None)
    if not callable(unregister):
        return
    for table_name in temp_tables:
        unregister(table_name)


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
        return self.relation.fetch_record_batch(batch_size)

    def to_table(self) -> pa.Table:
        """Execute the plan and return an Arrow table.

        Returns
        -------
        pyarrow.Table
            Materialized Arrow table.
        """
        return self.relation.fetch_arrow_table()

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
        return result.fetch_record_batch(batch_size)

    def to_table(self) -> pa.Table:
        """Execute the plan and return an Arrow table.

        Returns
        -------
        pyarrow.Table
            Materialized Arrow table.
        """
        result = self._execute()
        fetcher = getattr(result, "fetch_arrow_table", None)
        if callable(fetcher):
            return fetcher()
        reader = result.fetch_record_batch(10_000)
        return pa.Table.from_batches(list(reader), schema=reader.schema)

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
    """DuckDB engine backed by relations with Ibis fallback."""

    name: str = "duckdb"

    def can_run(self, spec: SemanticQuerySpec, *, ctx: EngineContext) -> bool:
        """Return True when DuckDB can satisfy the spec.

        Parameters
        ----------
        spec
            Semantic query spec to evaluate.
        ctx
            Engine context with warehouse access.

        Returns
        -------
        bool
            True if the engine can execute the spec.
        """
        return ctx.warehouse is not None and self.name.lower() == "duckdb" and bool(spec.table_key)

    def compile(self, spec: SemanticQuerySpec, *, ctx: EngineContext) -> ExecutablePlan:
        """Compile a semantic query spec into a DuckDB execution plan.

        Parameters
        ----------
        spec
            Semantic query spec to compile.
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
        try:
            relation = build_relation_plan(
                con=ctx.warehouse.gateway.con,
                spec=spec,
                dataset_manifests=ctx.dataset_manifests,
                column_types=spec.column_types,
            )
            return DuckDBRelationPlan(relation=relation, warehouse=ctx.warehouse)
        except DuckDBRelationQueryBuilderError as exc:
            relation_error = exc

        sqlglot_error: Exception | None = None
        try:
            sql_expr = build_sqlglot_query(
                spec=spec,
                allowed_columns=spec.allowed_columns,
                column_types=spec.column_types,
            )
            sql = render_sql_duckdb(sql_expr)
            policy = _sql_policy_for_spec(spec, temp_tables=())
            validated = _validate_sql(sql, policy=policy)
        except SqlglotQueryBuilderError as exc:
            sqlglot_error = exc
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

        ibis_con = ibis_facade.backend(ctx.warehouse.gateway)
        bound = build_query(ibis_con=ibis_con, spec=spec, column_types=spec.column_types)
        try:
            sql = bound.compile_sql(ibis_con)
            policy = _sql_policy_for_spec(spec, temp_tables=bound.temp_tables)
            validated = _validate_sql(sql, policy=policy)
        except UnsafeSqlError as exc:
            cleanup_temp_tables_if_needed(
                con=ctx.warehouse.gateway.con,
                temp_tables=bound.temp_tables,
            )
            raise QueryBuilderError(str(exc)) from exc
        except (IbisError, SqlglotError, TypeError, ValueError) as exc:
            cleanup_temp_tables_if_needed(
                con=ctx.warehouse.gateway.con,
                temp_tables=bound.temp_tables,
            )
            msg = f"DuckDB relation plan failed: {relation_error}; SQLGlot error: {sqlglot_error}"
            raise QueryBuilderError(msg) from exc
        return DuckDBSqlPlan(
            sql=validated,
            temp_tables=bound.temp_tables,
            warehouse=ctx.warehouse,
        )


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
