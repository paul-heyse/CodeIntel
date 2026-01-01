"""Metadata view definitions for the meta catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.storage.helpers.table_key import fully_qualified_table_ref
from codeintel.storage.sqlglot_tools import render_sql_duckdb, table_expr_from_ref

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

__all__ = ["apply_metadata_views"]


def _aliased_table(table_ref: str, alias: str) -> exp.Table:
    table_expr = table_expr_from_ref(table_ref)
    aliased = table_expr.copy()
    aliased.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
    return aliased


def _column_ref(column: str | tuple[str, str | None]) -> exp.Column:
    if isinstance(column, tuple):
        name, table = column
        if table:
            return exp.column(name, table=table)
        return exp.column(name)
    return exp.column(column)


def _row_number_expr(
    *,
    partition_by: list[str | tuple[str, str | None]],
    order_by: list[tuple[str | tuple[str, str | None], bool]],
    alias: str,
) -> exp.Expression:
    partitions = [_column_ref(column) for column in partition_by]
    orders = [
        exp.Ordered(this=_column_ref(column), desc=descending) for column, descending in order_by
    ]
    window = exp.Window(
        this=exp.RowNumber(),
        partition_by=partitions,
        order=exp.Order(expressions=orders),
    )
    return exp.alias_(window, alias)


def apply_metadata_views(con: DuckDBPyConnection, *, catalog: str | None) -> None:
    """Create or replace metadata views."""
    _apply_validation_summary_view(con, catalog=catalog)
    _apply_validation_failures_view(con, catalog=catalog)
    _apply_latest_good_view(con, catalog=catalog)


def _create_or_replace_view(
    con: DuckDBPyConnection,
    *,
    view_name: str,
    select_expr: exp.Expression,
) -> None:
    relation = con.sql(render_sql_duckdb(select_expr))
    create_view = getattr(relation, "create_view", None)
    if not callable(create_view):
        msg = "DuckDB relation does not support create_view"
        raise TypeError(msg)
    create_view(view_name, replace=True)


def _apply_validation_summary_view(con: DuckDBPyConnection, *, catalog: str | None) -> None:
    summary_view = fully_qualified_table_ref(
        "metadata.v_schema_validation_summary",
        catalog=catalog,
    )
    runs_ref = fully_qualified_table_ref(
        "metadata.schema_validation_runs",
        catalog=catalog,
    )
    summary_row_num = _row_number_expr(
        partition_by=["repo", "commit"],
        order_by=[("created_at", True)],
        alias="row_num",
    )
    summary_inner = exp.select(
        exp.column("validation_id"),
        exp.column("repo"),
        exp.column("commit"),
        exp.column("validation_mode"),
        exp.column("include_views"),
        exp.column("status"),
        exp.column("issue_count"),
        exp.column("created_at"),
        summary_row_num,
    ).from_(table_expr_from_ref(runs_ref))
    summary_ranked = exp.Subquery(
        this=summary_inner,
        alias=exp.TableAlias(this=exp.to_identifier("ranked")),
    )
    summary_outer = (
        exp.select(
            exp.column("validation_id"),
            exp.column("repo"),
            exp.column("commit"),
            exp.column("validation_mode"),
            exp.column("include_views"),
            exp.column("status"),
            exp.column("issue_count"),
            exp.column("created_at"),
        )
        .from_(summary_ranked)
        .where(exp.EQ(this=exp.column("row_num"), expression=exp.Literal.number(1)))
    )
    _create_or_replace_view(con, view_name=summary_view, select_expr=summary_outer)


def _apply_validation_failures_view(con: DuckDBPyConnection, *, catalog: str | None) -> None:
    failures_view = fully_qualified_table_ref(
        "metadata.v_schema_validation_failures",
        catalog=catalog,
    )
    runs_ref = fully_qualified_table_ref(
        "metadata.schema_validation_runs",
        catalog=catalog,
    )
    failures_row_num = _row_number_expr(
        partition_by=["repo", "commit"],
        order_by=[("created_at", True)],
        alias="row_num",
    )
    failures_inner = (
        exp.select(
            exp.column("validation_id"),
            exp.column("repo"),
            exp.column("commit"),
            exp.column("validation_mode"),
            exp.column("include_views"),
            exp.column("status"),
            exp.column("issue_count"),
            exp.column("issues"),
            exp.column("created_at"),
            failures_row_num,
        )
        .from_(table_expr_from_ref(runs_ref))
        .where(exp.EQ(this=exp.column("status"), expression=exp.Literal.string("failed")))
    )
    failures_ranked = exp.Subquery(
        this=failures_inner,
        alias=exp.TableAlias(this=exp.to_identifier("ranked")),
    )
    failures_outer = (
        exp.select(
            exp.column("validation_id"),
            exp.column("repo"),
            exp.column("commit"),
            exp.column("validation_mode"),
            exp.column("include_views"),
            exp.column("status"),
            exp.column("issue_count"),
            exp.column("issues"),
            exp.column("created_at"),
        )
        .from_(failures_ranked)
        .where(exp.EQ(this=exp.column("row_num"), expression=exp.Literal.number(1)))
    )
    _create_or_replace_view(con, view_name=failures_view, select_expr=failures_outer)


def _apply_latest_good_view(con: DuckDBPyConnection, *, catalog: str | None) -> None:
    latest_good_view = fully_qualified_table_ref(
        "metadata.v_schema_manifest_latest_good",
        catalog=catalog,
    )
    summary_view = fully_qualified_table_ref(
        "metadata.v_schema_validation_summary",
        catalog=catalog,
    )
    manifest_runs_ref = fully_qualified_table_ref(
        "metadata.schema_manifest_runs",
        catalog=catalog,
    )
    latest_row_num = _row_number_expr(
        partition_by=[("repo", "m")],
        order_by=[(("created_at", "m"), True)],
        alias="row_num",
    )
    manifest_table = _aliased_table(manifest_runs_ref, "m")
    summary_table = _aliased_table(summary_view, "s")
    join_condition = exp.and_(
        exp.EQ(this=exp.column("repo", table="m"), expression=exp.column("repo", table="s")),
        exp.EQ(this=exp.column("commit", table="m"), expression=exp.column("commit", table="s")),
    )
    latest_inner = (
        exp.select(
            exp.column("repo", table="m"),
            exp.column("commit", table="m"),
            exp.column("manifest_kind", table="m"),
            exp.column("catalog_hash", table="m"),
            exp.column("created_at", table="m"),
            exp.alias_(exp.column("status", table="s"), "validation_status"),
            exp.alias_(exp.column("created_at", table="s"), "validation_created_at"),
            latest_row_num,
        )
        .from_(manifest_table)
        .join(summary_table, on=join_condition)
        .where(
            exp.and_(
                exp.EQ(
                    this=exp.column("status", table="s"),
                    expression=exp.Literal.string("passed"),
                ),
                exp.GTE(
                    this=exp.column("created_at", table="s"),
                    expression=exp.column("created_at", table="m"),
                ),
            )
        )
    )
    latest_ranked = exp.Subquery(
        this=latest_inner,
        alias=exp.TableAlias(this=exp.to_identifier("ranked")),
    )
    latest_outer = (
        exp.select(
            exp.column("repo"),
            exp.column("commit"),
            exp.column("manifest_kind"),
            exp.column("catalog_hash"),
            exp.alias_(exp.column("created_at"), "manifest_created_at"),
            exp.column("validation_status"),
            exp.column("validation_created_at"),
        )
        .from_(latest_ranked)
        .where(exp.EQ(this=exp.column("row_num"), expression=exp.Literal.number(1)))
    )
    _create_or_replace_view(con, view_name=latest_good_view, select_expr=latest_outer)
