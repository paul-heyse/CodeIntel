"""Metadata bootstrap helpers.

This module contains low-level bulk insert helpers used by metadata sync.
It is intentionally the only metadata module that uses ``executemany`` so
bulk-write patterns remain centralized and easy to audit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.sqlglot_tools import render_sql_duckdb, table_expr_from_ref

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from duckdb import DuckDBPyConnection

__all__ = [
    "replace_dataset_dataflow_edges",
    "replace_dataset_dataflow_nodes",
    "replace_derived_lineage_columns",
    "replace_derived_lineage_edges",
]


def _delete_expr(
    table_ref: str,
    *,
    where_columns: Sequence[str] | None = None,
) -> exp.Delete:
    table_expr = table_expr_from_ref(table_ref)
    where_expr: exp.Expression | None = None
    if where_columns:
        for column in where_columns:
            comparison = exp.EQ(this=exp.column(column), expression=exp.Placeholder())
            where_expr = comparison if where_expr is None else exp.and_(where_expr, comparison)
    return exp.Delete(this=table_expr, where=where_expr)


def _insert_expr(table_ref: str, columns: Sequence[str]) -> exp.Insert:
    table_expr = table_expr_from_ref(table_ref)
    placeholders = [exp.Placeholder() for _ in columns]
    return exp.Insert(
        this=exp.Schema(
            this=table_expr,
            expressions=[exp.to_identifier(column) for column in columns],
        ),
        expression=exp.Values(expressions=[exp.Tuple(expressions=placeholders)]),
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
    table_ref = meta_table_ref("metadata.dataset_dataflow_nodes")
    delete_expr = _delete_expr(table_ref)
    con.execute(render_sql_duckdb(delete_expr))
    if not rows:
        return
    insert_expr = _insert_expr(
        table_ref,
        ["id", "kind", "family", "owner_package", "description"],
    )
    con.executemany(
        render_sql_duckdb(insert_expr),
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
    delete_expr = _delete_expr(table_ref)
    con.execute(render_sql_duckdb(delete_expr))
    if not rows:
        return
    insert_expr = _insert_expr(table_ref, ["src", "dst", "edge_type"])
    con.executemany(
        render_sql_duckdb(insert_expr),
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
    delete_expr = _delete_expr(
        table_ref,
        where_columns=["repo", "commit", "edge_type"],
    )
    con.execute(render_sql_duckdb(delete_expr), [repo, commit, edge_type])
    row_list = list(rows)
    if not row_list:
        return
    insert_expr = _insert_expr(
        table_ref,
        ["repo", "commit", "downstream", "upstream", "edge_type"],
    )
    con.executemany(
        render_sql_duckdb(insert_expr),
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
    delete_expr = _delete_expr(
        table_ref,
        where_columns=["repo", "commit", "edge_type"],
    )
    con.execute(render_sql_duckdb(delete_expr), [repo, commit, edge_type])
    row_list = list(rows)
    if not row_list:
        return
    insert_expr = _insert_expr(
        table_ref,
        [
            "repo",
            "commit",
            "downstream_table",
            "downstream_column",
            "upstream_table",
            "upstream_column",
            "edge_type",
        ],
    )
    con.executemany(
        render_sql_duckdb(insert_expr),
        row_list,
    )
