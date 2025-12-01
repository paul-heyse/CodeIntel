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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from codeintel.storage.gateway import DuckDBConnection, StorageGateway

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

    Fetches limit+1 rows to detect if more data exists beyond the page.

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
    >>> result.truncated  # True if more than 10 rows exist
    """
    # Fetch one extra row to detect truncation
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
