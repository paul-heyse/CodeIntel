"""Metadata bootstrap helpers.

This module contains low-level bulk insert helpers used by metadata sync.
It is intentionally the only metadata module that uses ``executemany`` so
bulk-write patterns remain centralized and easy to audit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from duckdb import DuckDBPyConnection

__all__ = [
    "replace_dataset_dataflow_edges",
    "replace_dataset_dataflow_nodes",
    "replace_dataset_schema_registry",
    "replace_derived_lineage_columns",
    "replace_derived_lineage_edges",
]


def replace_dataset_schema_registry(con: DuckDBPyConnection, *, entries: Mapping[str, str]) -> None:
    """Replace metadata.dataset_schema_registry with the provided entries.

    Parameters
    ----------
    con
        DuckDB connection.
    entries
        Mapping of table_key to schema hash.
    """
    con.execute("DELETE FROM metadata.dataset_schema_registry")
    con.executemany(
        """
        INSERT INTO metadata.dataset_schema_registry (table_key, schema_hash)
        VALUES (?, ?)
        """,
        list(entries.items()),
    )


def replace_dataset_dataflow_nodes(
    con: DuckDBPyConnection,
    *,
    rows: Sequence[tuple[str, str, str | None, str | None, str | None]],
) -> None:
    """Replace metadata.dataset_dataflow_nodes rows.

    Parameters
    ----------
    con
        DuckDB connection.
    rows
        Node rows in table order: (id, kind, family, owner_package, description).
    """
    con.execute("DELETE FROM metadata.dataset_dataflow_nodes")
    if not rows:
        return
    con.executemany(
        """
        INSERT INTO metadata.dataset_dataflow_nodes (
            id,
            kind,
            family,
            owner_package,
            description
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def replace_dataset_dataflow_edges(
    con: DuckDBPyConnection,
    *,
    rows: Sequence[tuple[str, str, str]],
) -> None:
    """Replace metadata.dataset_dataflow_edges rows.

    Parameters
    ----------
    con
        DuckDB connection.
    rows
        Edge rows in table order: (src, dst, edge_type).
    """
    con.execute("DELETE FROM metadata.dataset_dataflow_edges")
    if not rows:
        return
    con.executemany(
        """
        INSERT INTO metadata.dataset_dataflow_edges (
            src,
            dst,
            edge_type
        )
        VALUES (?, ?, ?)
        """,
        list(rows),
    )


def replace_derived_lineage_edges(
    con: DuckDBPyConnection,
    *,
    repo: str,
    commit: str,
    edge_type: str,
    rows: Iterable[tuple[str, str, str, str, str]],
) -> None:
    """Replace derived lineage edges for a snapshot.

    Parameters
    ----------
    con
        DuckDB connection.
    repo
        Repository identifier.
    commit
        Commit identifier.
    edge_type
        Edge type name.
    rows
        Row iterable in table order:
        (repo, commit, downstream, upstream, edge_type).
    """
    con.execute(
        """
        DELETE FROM metadata.derived_lineage_edges
        WHERE repo = ? AND commit = ? AND edge_type = ?
        """,
        [repo, commit, edge_type],
    )
    row_list = list(rows)
    if not row_list:
        return
    con.executemany(
        """
        INSERT INTO metadata.derived_lineage_edges (
            repo,
            commit,
            downstream,
            upstream,
            edge_type
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        row_list,
    )


def replace_derived_lineage_columns(
    con: DuckDBPyConnection,
    *,
    repo: str,
    commit: str,
    edge_type: str,
    rows: Iterable[tuple[str, str, str, str, str, str, str]],
) -> None:
    """Replace derived lineage column edges for a snapshot.

    Parameters
    ----------
    con
        DuckDB connection.
    repo
        Repository identifier.
    commit
        Commit identifier.
    edge_type
        Edge type name.
    rows
        Row iterable in table order:
        (repo, commit, downstream_table, downstream_column,
         upstream_table, upstream_column, edge_type).
    """
    con.execute(
        """
        DELETE FROM metadata.derived_lineage_columns
        WHERE repo = ? AND commit = ? AND edge_type = ?
        """,
        [repo, commit, edge_type],
    )
    row_list = list(rows)
    if not row_list:
        return
    con.executemany(
        """
        INSERT INTO metadata.derived_lineage_columns (
            repo,
            commit,
            downstream_table,
            downstream_column,
            upstream_table,
            upstream_column,
            edge_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        row_list,
    )
