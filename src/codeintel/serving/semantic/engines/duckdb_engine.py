"""DuckDB/Ibis-based semantic query engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.serving.semantic.engines.protocol import (
    EngineContext,
    ExecutablePlan,
    QueryExplain,
)
from codeintel.serving.semantic.query_builder import QueryBuilderError, build_query
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.storage.gateway import ibis_facade
from codeintel.storage.queries.safe import SqlIngressPolicy, UnsafeSqlError, assert_select_perimeter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection


def _format_explain_rows(rows: Sequence[Sequence[object]]) -> str:
    plan_lines: list[str] = []
    for row in rows:
        if not row:
            continue
        plan_lines.append(str(row[1] if len(row) > 1 else row[0]))
    return "\n".join(plan_lines)


def cleanup_temp_tables_if_needed(*, con: DuckDBPyConnection, temp_tables: Sequence[str]) -> None:
    """Unregister Ibis temp tables when supported by the backend."""
    unregister = getattr(con, "unregister", None)
    if not callable(unregister):
        return
    for table_name in temp_tables:
        unregister(table_name)


@dataclass(frozen=True, slots=True)
class DuckDBExecutablePlan:
    """Executable DuckDB plan wrapper."""

    sql: str
    temp_tables: tuple[str, ...]
    warehouse: object

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Execute the plan and return a RecordBatchReader.

        Parameters
        ----------
        batch_size
            Max batch size per chunk in the reader.

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
            Explain payload containing SQL and plan text.
        """
        result = self._execute_sql(f"EXPLAIN {self.sql}")
        plan_text = _format_explain_rows(result.fetchall())
        return QueryExplain(sql=self.sql, plan=plan_text)

    def cleanup(self) -> None:
        """Release temporary resources after execution."""
        con = getattr(self.warehouse, "gateway", None)
        conn = getattr(con, "con", None) if con is not None else None
        if conn is None:
            return
        cleanup_temp_tables_if_needed(con=conn, temp_tables=self.temp_tables)

    def _execute(self) -> DuckDBPyConnection:
        return self._execute_sql(self.sql)

    def _execute_sql(self, sql: str) -> DuckDBPyConnection:
        gateway = getattr(self.warehouse, "gateway", None)
        if gateway is None:
            msg = "DuckDBExecutablePlan requires a warehouse gateway"
            raise QueryBuilderError(msg)
        return gateway.policy.execute_sql(sql)


@dataclass(frozen=True, slots=True)
class DuckDBQueryEngine:
    """DuckDB engine backed by the Ibis query builder."""

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
        ibis_con = ibis_facade.backend(ctx.warehouse.gateway)
        bound = build_query(ibis_con=ibis_con, spec=spec, column_types=spec.column_types)
        try:
            sql = bound.compile_sql(ibis_con)
            assert_select_perimeter(sql, policy=SqlIngressPolicy())
        except UnsafeSqlError as exc:
            cleanup_temp_tables_if_needed(
                con=ctx.warehouse.gateway.con,
                temp_tables=bound.temp_tables,
            )
            raise QueryBuilderError(str(exc)) from exc
        except Exception:
            cleanup_temp_tables_if_needed(
                con=ctx.warehouse.gateway.con,
                temp_tables=bound.temp_tables,
            )
            raise
        return DuckDBExecutablePlan(
            sql=sql,
            temp_tables=bound.temp_tables,
            warehouse=ctx.warehouse,
        )


__all__ = ["DuckDBExecutablePlan", "DuckDBQueryEngine", "cleanup_temp_tables_if_needed"]
