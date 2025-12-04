"""Data existence checking utilities for the storage layer.

This module provides functions to check whether dataset tables contain
data for specific repository/commit snapshots. These utilities are used
by the auto-pipeline to determine if prerequisite data exists.

The functions in this module are designed to be efficient, using LIMIT 1
queries to minimize database overhead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.datasets import DatasetContract
    from codeintel.storage.gateway import DuckDBConnection

LOG = logging.getLogger(__name__)

__all__ = [
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
