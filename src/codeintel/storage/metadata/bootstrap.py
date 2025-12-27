"""Metadata bootstrap helpers.

This module contains low-level bulk insert helpers used by metadata sync.
It is intentionally the only metadata module that uses ``executemany`` so
bulk-write patterns remain centralized and easy to audit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.metadata.meta_catalog import meta_table_ref

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from duckdb import DuckDBPyConnection

__all__ = [
    "replace_dataset_dataflow_edges",
    "replace_dataset_dataflow_nodes",
    "replace_derived_lineage_columns",
    "replace_derived_lineage_edges",
]


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
    table_ref = meta_table_ref("metadata.dataset_dataflow_nodes")
    con.execute(f"DELETE FROM {table_ref}")
    if not rows:
        return
    con.executemany(
        f"""
        INSERT INTO {table_ref} (
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
    table_ref = meta_table_ref("metadata.dataset_dataflow_edges")
    con.execute(f"DELETE FROM {table_ref}")
    if not rows:
        return
    con.executemany(
        f"""
        INSERT INTO {table_ref} (
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
    table_ref = meta_table_ref("metadata.derived_lineage_edges")
    con.execute(
        f"""
        DELETE FROM {table_ref}
        WHERE repo = ? AND commit = ? AND edge_type = ?
        """,
        [repo, commit, edge_type],
    )
    row_list = list(rows)
    if not row_list:
        return
    con.executemany(
        f"""
        INSERT INTO {table_ref} (
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
    table_ref = meta_table_ref("metadata.derived_lineage_columns")
    con.execute(
        f"""
        DELETE FROM {table_ref}
        WHERE repo = ? AND commit = ? AND edge_type = ?
        """,
        [repo, commit, edge_type],
    )
    row_list = list(rows)
    if not row_list:
        return
    con.executemany(
        f"""
        INSERT INTO {table_ref} (
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
