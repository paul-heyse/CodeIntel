"""Data existence checking utilities for the storage layer.

This module provides functions to check whether dataset tables contain
data for specific repository/commit snapshots. These utilities are used
by the auto-pipeline to determine if prerequisite data exists.

The functions in this module are designed to be efficient, using LIMIT 1
queries to minimize database overhead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from duckdb import DuckDBPyConnection
from duckdb import Error as DuckDBError

if TYPE_CHECKING:
    from codeintel.config.datasets import DatasetContract

# Type alias for DuckDB connection
DuckDBConnection = DuckDBPyConnection

LOG = logging.getLogger(__name__)

__all__ = [
    "count_rows_for_snapshot",
    "count_rows_for_tables",
    "safe_count_rows",
    "table_has_rows_for_snapshot",
]


def table_has_rows_for_snapshot(
    con: DuckDBConnection,
    contract: DatasetContract,
    *,
    repo: str,
    commit: str,
) -> bool:
    """Check if a dataset table has rows for the given repo/commit.

    Performs a LIMIT 1 query to efficiently check for data existence
    without fetching all rows. Uses the DuckDB relation API for safe
    table name handling.

    Parameters
    ----------
    con
        DuckDB connection.
    contract
        Dataset contract with schema information.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    bool
        True if at least one row exists, False otherwise.

    Notes
    -----
    If the contract schema has both 'repo' and 'commit' columns, the query
    filters by those values. Otherwise, it just checks for any row existence.
    Database errors are caught and logged, returning False on failure.
    """
    table_key = contract.table_key
    schema = contract.schema
    has_repo_col = schema is not None and any(c.name == "repo" for c in schema.columns)
    has_commit_col = schema is not None and any(c.name == "commit" for c in schema.columns)

    try:
        # Use the relation API for safe table name handling
        relation = con.table(table_key)

        if has_repo_col and has_commit_col:
            # Escape single quotes in repo/commit values
            escaped_repo = repo.replace("'", "''")
            escaped_commit = commit.replace("'", "''")
            relation = relation.filter(f"repo = '{escaped_repo}' AND commit = '{escaped_commit}'")

        result = relation.limit(1).fetchone()
    except (RuntimeError, ValueError, OSError) as exc:
        LOG.debug(
            "table_has_rows_for_snapshot: error checking %s: %s",
            table_key,
            exc,
        )
        return False
    else:
        return result is not None


def count_rows_for_snapshot(
    con: DuckDBConnection,
    table_key: str,
    *,
    repo: str,
    commit: str,
) -> int:
    """Count rows in a table filtered by repo/commit.

    Parameters
    ----------
    con
        DuckDB connection.
    table_key
        Fully qualified table name (schema.table).
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of rows matching the repo/commit filter.
    """
    escaped_repo = repo.replace("'", "''")
    escaped_commit = commit.replace("'", "''")
    relation = con.table(table_key).filter(
        f"repo = '{escaped_repo}' AND commit = '{escaped_commit}'"
    )
    result = relation.count("*").fetchone()
    if result is None:
        return 0
    return int(result[0])


def count_rows_for_tables(
    con: DuckDBConnection,
    tables: Sequence[str],
    *,
    repo: str,
    commit: str,
) -> dict[str, int] | None:
    """Compute row counts for multiple tables filtered by repo/commit.

    Parameters
    ----------
    con
        DuckDB connection.
    tables
        Sequence of fully qualified table names.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    dict[str, int] | None
        Mapping of table name to row counts, or None if any table fails.
    """
    counts: dict[str, int] = {}
    for table in tables:
        try:
            counts[table] = count_rows_for_snapshot(con, table, repo=repo, commit=commit)
        except DuckDBError:
            return None
    return counts


def safe_count_rows(
    con: DuckDBConnection | None,
    tables: Iterable[str],
    *,
    repo: str,
    commit: str,
) -> dict[str, int] | None:
    """Tolerant variant of count_rows_for_tables that handles None connection.

    Parameters
    ----------
    con
        DuckDB connection, or None.
    tables
        Iterable of fully qualified table names.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    dict[str, int] | None
        Row counts, or None when connection is unavailable or query fails.
    """
    if con is None:
        return None
    return count_rows_for_tables(con, tuple(tables), repo=repo, commit=commit)
