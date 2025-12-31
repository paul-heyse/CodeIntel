"""Typed adapters for DuckDB relation scans."""

from __future__ import annotations

from collections.abc import Sequence

from codeintel.storage.duckdb_types import DuckDBConnection, DuckDBRelation


def scan_parquet(
    con: DuckDBConnection,
    *,
    scan_paths: Sequence[str],
    hive_partitioning: bool,
    union_by_name: bool,
    columns: Sequence[str] | None = None,
) -> DuckDBRelation:
    """Return a DuckDB relation for Parquet paths.

    Returns
    -------
    DuckDBRelation
        DuckDB relation scanning the provided Parquet paths.
    """
    relation = con.from_parquet(
        list(scan_paths),
        hive_partitioning=hive_partitioning,
        union_by_name=union_by_name,
    )
    if columns is None:
        return relation
    return relation.select(*columns)


def scan_arrow(con: DuckDBConnection, *, source: object) -> DuckDBRelation:
    """Return a DuckDB relation for an Arrow-backed source.

    Returns
    -------
    DuckDBRelation
        DuckDB relation scanning the provided Arrow source.
    """
    return con.from_arrow(source)
