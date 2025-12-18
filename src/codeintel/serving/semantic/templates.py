"""Typed query templates for semantic execution.

This module defines a dual-mode abstraction for executing semantic queries:

- Ibis-first templates (`QueryTemplate` + `BoundQuery`) for safe, typed query building.
- DB-API templates (`DbApiQuery`) for existing raw SQL hot paths (e.g., search).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import ibis.expr.types as it
    from ibis.backends.duckdb import Backend as DuckDBBackend


@dataclass(frozen=True, slots=True)
class QueryTemplate:
    """A reusable semantic query shape (Ibis expression + temp resources)."""

    expr: it.Table
    temp_tables: tuple[str, ...] = ()

    def bind(self, params: Mapping[it.Expr, object]) -> BoundQuery:
        """Bind parameters for execution.

        Parameters
        ----------
        params
            Mapping of Ibis parameter expressions (typically ``ibis.param(...)`` scalars)
            to their bound runtime values.

        Returns
        -------
        BoundQuery
            Bound query instance ready for compilation/execution.
        """
        return BoundQuery(template=self, params=dict(params))


@dataclass(frozen=True, slots=True)
class BoundQuery:
    """A query template with bound scalar parameter values."""

    template: QueryTemplate
    params: dict[it.Expr, object]

    @property
    def expr(self) -> it.Table:
        """Return the underlying Ibis expression.

        Returns
        -------
        ibis.expr.types.Table
            Table expression.
        """
        return self.template.expr

    @property
    def temp_tables(self) -> tuple[str, ...]:
        """Return staged temporary table names, if any.

        Returns
        -------
        tuple[str, ...]
            Temporary DuckDB table names that must be cleaned up after execution.
        """
        return self.template.temp_tables

    def execute_params(self) -> Mapping[it.Value, object]:
        """Return params mapping typed for ``Expr.execute(params=...)``.

        Returns
        -------
        Mapping[ibis.expr.types.Value, object]
            Mapping compatible with ``execute(params=...)``.
        """
        return cast("Mapping[it.Value, object]", self.params)

    def compile_sql(self, ibis_con: DuckDBBackend) -> str:
        """Compile to DuckDB SQL with parameters safely embedded by Ibis.

        Parameters
        ----------
        ibis_con
            Ibis backend bound to the target DuckDB connection.

        Returns
        -------
        str
            Compiled DuckDB SQL string.
        """
        if not self.params:
            return ibis_con.compile(self.expr)
        return ibis_con.compile(self.expr, params=self.params)


@dataclass(frozen=True, slots=True)
class DbApiQuery:
    """Raw SQL + positional parameters for DB-API execution."""

    sql: str
    params: Sequence[object] | None = None


@dataclass(frozen=True, slots=True)
class DbApiTemplate:
    """Reusable DB-API query shape (SQL string + params bound at runtime)."""

    sql: str

    def bind(self, params: Sequence[object] | None = None) -> DbApiQuery:
        """Bind positional parameters for DB-API execution.

        Parameters
        ----------
        params
            Positional parameter values for ``?`` placeholders in ``sql``.

        Returns
        -------
        DbApiQuery
            Bound query ready for DB-API execution.
        """
        return DbApiQuery(sql=self.sql, params=params)


__all__ = [
    "BoundQuery",
    "DbApiQuery",
    "DbApiTemplate",
    "QueryTemplate",
]
