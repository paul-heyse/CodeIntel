"""SQLGlot tombstone filtering helpers for serving."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlglot import exp

from codeintel.core.columnar.tabular_adapter import PolarsLazyFrame, to_lazyframe
from codeintel.serving.semantic.iceberg_scans import (
    IcebergScanError,
    IcebergScanRequest,
    iceberg_scan_for_query,
)
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.core.config.settings import IcebergSettings
    from codeintel.serving.db.pointer import ServingSnapshotPointer

try:
    import polars as pl
    from polars.exceptions import PolarsError
except ImportError:  # pragma: no cover
    pl = None
    PolarsError = Exception

LOG = logging.getLogger(__name__)


def apply_tombstone_filter(
    ast: exp.Select,
    *,
    table_key: str,
    primary_key: Sequence[str],
    snapshot_id: int | None,
) -> exp.Select:
    """Apply a NOT EXISTS tombstone filter to a simple SELECT.

    Returns
    -------
    sqlglot.expressions.Select
        Updated AST with tombstone filter applied when eligible.
    """
    if not primary_key:
        return ast
    if _has_joins(ast):
        return ast
    base_table = _base_table(ast)
    if base_table is None:
        return ast
    tombstone_table_key = _tombstone_table_key(table_key)
    if _contains_table(ast, tombstone_table_key):
        return ast
    base_alias = base_table.alias_or_name
    tombstone_expr, tombstone_alias = _tombstone_table_expr(tombstone_table_key)
    predicates: list[exp.Expression] = [
        exp.EQ(
            this=exp.column(col, table=base_alias),
            expression=exp.column(col, table=tombstone_alias),
        )
        for col in primary_key
    ]
    if snapshot_id is not None:
        predicates.append(
            exp.LTE(
                this=exp.column("snapshot_id", table=tombstone_alias),
                expression=exp.Literal.number(snapshot_id),
            )
        )
    if not predicates:
        return ast
    not_exists = exp.Not(
        this=exp.Exists(
            this=exp.select("1").from_(tombstone_expr).where(_combine_predicates(predicates))
        )
    )
    cloned = ast.copy()
    return cloned.where(not_exists)


def apply_tombstone_filter_lazyframe(
    lazyframe: PolarsLazyFrame,
    *,
    table_key: str,
    primary_key: Sequence[str],
    snapshot_id: int | None,
    pointer: ServingSnapshotPointer,
    settings: IcebergSettings,
    batch_size: int | None,
) -> PolarsLazyFrame:
    """Apply tombstone filtering to a Polars LazyFrame."""
    if (
        not settings.tombstones_enabled
        or not primary_key
        or snapshot_id is None
        or pl is None  # pragma: no cover
    ):
        return lazyframe
    tombstone_key = _tombstone_table_key(table_key)
    try:
        scan_result = iceberg_scan_for_query(
            request=IcebergScanRequest(
                table_key=tombstone_key,
                columns=(*tuple(primary_key), "snapshot_id"),
                filters=[],
                order_by=[],
                column_types=None,
                pointer=pointer,
                settings=settings,
                batch_size=batch_size,
            )
        )
    except IcebergScanError as exc:
        LOG.warning("Tombstone scan failed for %s: %s", tombstone_key, exc)
        return lazyframe
    tombstones = to_lazyframe(scan_result.scan)
    try:
        tombstones = tombstones.filter(pl.col("snapshot_id") <= snapshot_id)
        joined = lazyframe.join(tombstones, on=list(primary_key), how="anti")
    except PolarsError as exc:
        LOG.warning("Polars tombstone anti-join failed: %s", exc)
        return lazyframe
    if isinstance(joined, pl.LazyFrame):
        return joined
    return lazyframe


def _base_table(ast: exp.Select) -> exp.Table | None:
    from_expr = ast.args.get("from_")
    if not isinstance(from_expr, exp.From):
        return None
    if ast.args.get("joins"):
        return None
    table_expr = from_expr.this
    if isinstance(table_expr, exp.Table):
        return table_expr
    return None


def _has_joins(ast: exp.Select) -> bool:
    return bool(ast.args.get("joins"))


def _tombstone_table_key(table_key: str) -> str:
    schema, table = split_table_key(table_key)
    return f"{schema}.{table}__tombstones"


def _tombstone_table_expr(table_key: str) -> tuple[exp.Table, str]:
    schema, table = split_table_key(table_key)
    table_expr = exp.Table(
        this=exp.to_identifier(table),
        db=exp.to_identifier(schema),
    )
    return table_expr, table


def _contains_table(ast: exp.Select, table_key: str) -> bool:
    schema, table = split_table_key(table_key)
    return any(node.name == table and node.db == schema for node in ast.find_all(exp.Table))


def _combine_predicates(predicates: Sequence[exp.Expression]) -> exp.Expression:
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = exp.and_(combined, predicate)
    return combined


__all__ = ["apply_tombstone_filter", "apply_tombstone_filter_lazyframe"]
