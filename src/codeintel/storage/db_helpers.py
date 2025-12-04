"""DuckDB-specific helpers scoped to the storage layer."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from codeintel.storage.gateway import DuckDBConnection, DuckDBError

DUCKDB_ERRORS: tuple[type[Exception], ...] = (DuckDBError,)


def row_counts_for_tables(
    con: DuckDBConnection,
    *,
    repo: str,
    commit: str,
    tables: Sequence[str],
) -> dict[str, int] | None:
    """
    Compute row counts for each table filtered by repo/commit.

    Returns
    -------
    dict[str, int] | None
        Mapping of table name to counts, or None if any table fails to count.
    """
    counts: dict[str, int] = {}
    for table in tables:
        try:
            escaped_repo = repo.replace("'", "''")
            escaped_commit = commit.replace("'", "''")
            relation = con.table(table).filter(
                f"repo = '{escaped_repo}' AND commit = '{escaped_commit}'"
            )
            result = relation.count("*").fetchone()
            if result is None:
                return None
            counts[table] = int(result[0])
        except DuckDBError:
            return None
    return counts


def safe_row_counts(
    con: DuckDBConnection | None,
    *,
    repo: str,
    commit: str,
    tables: Iterable[str],
) -> dict[str, int] | None:
    """
    Variant that tolerates missing connection or empty tables.

    Returns
    -------
    dict[str, int] | None
        Row counts or None when unavailable.
    """
    if con is None:
        return None
    return row_counts_for_tables(con, repo=repo, commit=commit, tables=tuple(tables))
