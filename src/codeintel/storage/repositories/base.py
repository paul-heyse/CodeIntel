"""Shared repository helpers for DuckDB-backed storage.

This module provides base classes and helper functions for DuckDB-backed
repositories. All repositories should extend BaseRepository and use the
standardized fetch helpers for consistent data access patterns.

Method Signature Patterns
-------------------------
Repositories should follow these patterns for method signatures:

Single-row fetch:
    def get_X(self, id: int) -> RowDict | None: ...

List fetch with pagination:
    def list_X(self, *, limit: int | None = None) -> list[RowDict]: ...

Paginated fetch with truncation detection:
    def list_X_paginated(self, *, limit: int) -> PaginatedRows: ...

Existence check:
    def has_X(self, id: int) -> bool: ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from ibis.common.exceptions import IbisError

from codeintel.config.datasets.validation import validate_df

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ibis.expr import types as it

    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

log = logging.getLogger(__name__)

RowDict = dict[str, Any]


@dataclass(frozen=True)
class PaginatedRows:
    """
    Result of a paginated repository fetch with truncation metadata.

    This type mirrors the serving layer's PaginatedFetch but is designed
    for use within the repository layer. It provides consistent truncation
    detection for all list operations.

    Parameters
    ----------
    rows
        The fetched rows (may be truncated if limit was applied).
    limit
        The limit used for the query.
    truncated
        Whether more rows exist beyond the limit.
    total_available
        Total row count if known (requires separate COUNT query).
    """

    rows: list[RowDict]
    limit: int
    truncated: bool
    total_available: int | None = None

    @property
    def count(self) -> int:
        """Return the number of rows in this page."""
        return len(self.rows)


def fetch_one_dict(con: DuckDBConnection, sql: str, params: Sequence[object]) -> RowDict | None:
    """
    Execute a query and return the first row as a mapping.

    .. deprecated::
        Use ``BaseRepository._ibis_to_one()`` instead. Raw SQL helpers will
        be removed in a future release. Migrate to Ibis-based queries.

    Parameters
    ----------
    con
        DuckDB connection.
    sql
        SQL query to execute.
    params
        Query parameters.

    Returns
    -------
    RowDict | None
        Mapping of column to value when a row exists; otherwise ``None``.
    """
    result = con.execute(sql, list(params))
    row = result.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in result.description]
    return {col: row[idx] for idx, col in enumerate(cols)}


def fetch_all_dicts(con: DuckDBConnection, sql: str, params: Sequence[object]) -> list[RowDict]:
    """
    Execute a query and return all rows as mappings.

    .. deprecated::
        Use ``BaseRepository._ibis_to_dicts()`` instead. Raw SQL helpers will
        be removed in a future release. Migrate to Ibis-based queries.

    Parameters
    ----------
    con
        DuckDB connection.
    sql
        SQL query to execute.
    params
        Query parameters.

    Returns
    -------
    list[RowDict]
        List of rows represented as dictionaries keyed by column name.
    """
    result = con.execute(sql, list(params))
    rows = result.fetchall()
    cols = [desc[0] for desc in result.description]
    return [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]


def fetch_paginated(
    con: DuckDBConnection,
    sql: str,
    params: Sequence[object],
    *,
    limit: int,
) -> PaginatedRows:
    """
    Execute a query with pagination and truncation detection.

    Fetch limit+1 rows to detect if more data exists beyond the page.

    .. deprecated::
        Use ``BaseRepository._ibis_paginated()`` instead. Raw SQL helpers will
        be removed in a future release. Migrate to Ibis-based queries.

    Parameters
    ----------
    con
        DuckDB connection.
    sql
        SQL query to execute (should include LIMIT placeholder).
    params
        Query parameters (limit will be appended).
    limit
        Maximum rows to return.

    Returns
    -------
    PaginatedRows
        Paginated result with truncation metadata.

    Examples
    --------
    >>> sql = "SELECT * FROM table WHERE x = ? LIMIT ?"
    >>> result = fetch_paginated(con, sql, [value], limit=10)
    >>> result.truncated
    """
    fetch_limit = limit + 1
    result = con.execute(sql, [*params, fetch_limit])
    all_rows = result.fetchall()
    cols = [desc[0] for desc in result.description]

    truncated = len(all_rows) > limit
    rows = [{col: row[idx] for idx, col in enumerate(cols)} for row in all_rows[:limit]]

    return PaginatedRows(rows=rows, limit=limit, truncated=truncated)


def row_exists(con: DuckDBConnection, sql: str, params: Sequence[object]) -> bool:
    """
    Check if at least one row matches the query.

    .. deprecated::
        Use ``BaseRepository._ibis_exists()`` instead. Raw SQL helpers will
        be removed in a future release. Migrate to Ibis-based queries.

    Parameters
    ----------
    con
        DuckDB connection.
    sql
        SQL query to execute.
    params
        Query parameters.

    Returns
    -------
    bool
        True if at least one row matches.
    """
    result = con.execute(sql, list(params))
    return result.fetchone() is not None


@dataclass(frozen=True)
class BaseRepository:
    """
    Base class for repositories bound to a gateway/revision.

    All repositories should extend this class to ensure consistent
    connection management and revision binding.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection.
    repo
        Repository identifier (e.g., "org/repo").
    commit
        Commit hash for the snapshot.
    """

    gateway: StorageGateway
    repo: str
    commit: str

    @property
    def con(self) -> DuckDBConnection:
        """Return the underlying DuckDB connection."""
        return self.gateway.con

    def _fetch_one(self, sql: str, params: Sequence[object]) -> RowDict | None:
        """
        Execute a query and return the first row.

        Parameters
        ----------
        sql
            SQL query to execute.
        params
            Query parameters.

        Returns
        -------
        RowDict | None
            First row as a dict, or None if no rows match.
        """
        return fetch_one_dict(self.con, sql, params)

    def _fetch_all(self, sql: str, params: Sequence[object]) -> list[RowDict]:
        """
        Execute a query and return all rows.

        Parameters
        ----------
        sql
            SQL query to execute.
        params
            Query parameters.

        Returns
        -------
        list[RowDict]
            All matching rows as dicts.
        """
        return fetch_all_dicts(self.con, sql, params)

    def _fetch_paginated(
        self,
        sql: str,
        params: Sequence[object],
        *,
        limit: int,
    ) -> PaginatedRows:
        """
        Execute a paginated query with truncation detection.

        Parameters
        ----------
        sql
            SQL query (should include LIMIT placeholder at end).
        params
            Query parameters (limit appended automatically).
        limit
            Maximum rows to return.

        Returns
        -------
        PaginatedRows
            Paginated result with truncation metadata.
        """
        return fetch_paginated(self.con, sql, params, limit=limit)

    def _exists(self, sql: str, params: Sequence[object]) -> bool:
        """
        Check if at least one row matches.

        Parameters
        ----------
        sql
            SQL query to execute.
        params
            Query parameters.

        Returns
        -------
        bool
            True if at least one row matches.
        """
        return row_exists(self.con, sql, params)

    def _ibis_table(self, table_key: str) -> it.Table:
        """
        Return an Ibis table expression for the given table key.

        Parameters
        ----------
        table_key
            Fully qualified table name (e.g., "core.goids").

        Returns
        -------
        it.Table
            Ibis table expression.
        """
        return self.gateway.ibis.table(table_key)

    def _ibis_to_df(
        self,
        expr: it.Table,
        table_key: str | None = None,
    ) -> pd.DataFrame:
        """
        Execute an Ibis expression and return a validated DataFrame.

        Parameters
        ----------
        expr
            Ibis table expression.
        table_key
            Optional table key for Pandera validation.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame.
        """
        _ = self
        df = pd.DataFrame(expr.execute())
        if table_key:
            return validate_df(table_key, df)
        return df

    def _ibis_to_dicts(
        self,
        expr: it.Table,
        table_key: str | None = None,
    ) -> list[RowDict]:
        """
        Execute an Ibis expression and return rows as dicts.

        Parameters
        ----------
        expr
            Ibis table expression.
        table_key
            Optional table key for Pandera validation.

        Returns
        -------
        list[RowDict]
            List of row dictionaries.
        """
        df = self._ibis_to_df(expr, table_key)
        sanitized = df.astype("object").where(pd.notna(df), None)
        return sanitized.to_dict(orient="records")

    def _ibis_to_one(
        self,
        expr: it.Table,
        table_key: str | None = None,
    ) -> RowDict | None:
        """
        Execute an Ibis expression and return the first row.

        Parameters
        ----------
        expr
            Ibis table expression.
        table_key
            Optional table key for Pandera validation.

        Returns
        -------
        RowDict | None
            First row as dict, or None if no rows.
        """
        dicts = self._ibis_to_dicts(expr.limit(1), table_key)
        return dicts[0] if dicts else None

    def _ibis_with_fallback(
        self,
        ibis_fn: Callable[[], it.Table],
        sql_fallback: str,
        params: Sequence[object],
        *,
        table_key: str | None = None,
    ) -> list[RowDict]:
        """
        Execute Ibis query with SQL fallback on error.

        Parameters
        ----------
        ibis_fn
            Callable returning an Ibis expression.
        sql_fallback
            SQL query to execute if Ibis fails.
        params
            Parameters for the SQL fallback.
        table_key
            Optional table key for Pandera validation.

        Returns
        -------
        list[RowDict]
            Query results as dictionaries.

        .. deprecated::
            Use ``_ibis_to_dicts`` directly. SQL fallbacks will be removed.
        """
        try:
            expr = ibis_fn()
            return self._ibis_to_dicts(expr, table_key)
        except IbisError:
            log.debug("Falling back to SQL for query")
            return self._fetch_all(sql_fallback, params)

    def _ibis_one_with_fallback(
        self,
        ibis_fn: Callable[[], it.Table],
        sql_fallback: str,
        params: Sequence[object],
        *,
        table_key: str | None = None,
    ) -> RowDict | None:
        """
        Execute Ibis query for single row with SQL fallback.

        Parameters
        ----------
        ibis_fn
            Callable returning an Ibis expression.
        sql_fallback
            SQL query to execute if Ibis fails.
        params
            Parameters for the SQL fallback.
        table_key
            Optional table key for Pandera validation.

        Returns
        -------
        RowDict | None
            Query result or None.

        .. deprecated::
            Use ``_ibis_to_one`` directly. SQL fallbacks will be removed.
        """
        try:
            expr = ibis_fn()
            return self._ibis_to_one(expr, table_key)
        except IbisError:
            log.debug("Falling back to SQL for single-row query")
            return self._fetch_one(sql_fallback, params)

    def _ibis_exists(
        self,
        expr: it.Table,
    ) -> bool:
        """
        Check if at least one row exists in the Ibis expression result.

        Execute an Ibis expression and return True if at least one row
        is present. This is the Ibis equivalent of ``_exists()``.

        Parameters
        ----------
        expr
            Ibis table expression (typically with filters applied).

        Returns
        -------
        bool
            True if at least one row matches the expression.
        """
        _ = self

        limited = expr.limit(1)
        df = pd.DataFrame(limited.execute())
        return len(df) > 0

    def _ibis_paginated(
        self,
        expr: it.Table,
        *,
        limit: int,
        table_key: str | None = None,
    ) -> PaginatedRows:
        """
        Execute an Ibis expression with pagination and truncation detection.

        Fetch limit+1 rows to detect if more data exists beyond the page,
        returning a PaginatedRows result with truncation metadata.

        Parameters
        ----------
        expr
            Ibis table expression (filters/ordering should already be applied).
        limit
            Maximum rows to return.
        table_key
            Optional table key for Pandera validation.

        Returns
        -------
        PaginatedRows
            Paginated result with truncation metadata.
        """
        _ = self

        fetch_limit = limit + 1
        limited_expr = expr.limit(fetch_limit)
        df = pd.DataFrame(limited_expr.execute())

        if table_key:
            df = validate_df(table_key, df)

        all_rows = df.to_dict(orient="records")
        truncated = len(all_rows) > limit
        rows = all_rows[:limit]

        return PaginatedRows(rows=rows, limit=limit, truncated=truncated)
