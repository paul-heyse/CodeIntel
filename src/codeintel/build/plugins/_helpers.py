"""Shared helper utilities for build plugins.

Provides common functionality used across multiple plugin implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.context import TargetExecutionContext

__all__ = [
    "compute_row_counts",
]


def compute_row_counts(
    ctx: TargetExecutionContext,
    table_keys: Iterable[str] | None = None,
) -> dict[str, int]:
    """Compute row counts for output tables in the current snapshot.

    Count rows in the specified tables that match the current repository
    and commit from the execution context. Handles missing tables gracefully.

    Parameters
    ----------
    ctx
        Execution context with gateway and snapshot.
    table_keys
        Specific table keys to count. If None, uses table keys from
        ``ctx.contract.table_keys``.

    Returns
    -------
    dict[str, int]
        Mapping of table key to row count. Returns 0 for tables that
        don't exist or cannot be queried.

    Examples
    --------
    Count rows for all contract tables:

    >>> counts = compute_row_counts(ctx)
    >>> print(counts)
    {'analytics.function_metrics': 42, 'analytics.type_coverage': 15}

    Count rows for specific tables:

    >>> counts = compute_row_counts(ctx, ['analytics.function_metrics'])
    >>> print(counts['analytics.function_metrics'])
    42
    """
    # Lazy imports to avoid circular dependencies
    from ibis.common.exceptions import IbisError  # noqa: PLC0415

    from codeintel.storage.gateway.protocol import DuckDBCatalogException  # noqa: PLC0415
    from codeintel.storage.ibis_types import filter_by, ibis_bool  # noqa: PLC0415

    keys = list(table_keys) if table_keys is not None else list(ctx.contract.table_keys)
    row_counts: dict[str, int] = {}

    for table_key in keys:
        try:
            table = ctx.gateway.ibis.table(table_key)
            count_expr = filter_by(
                table,
                ibis_bool(table.repo == ctx.repo),
                ibis_bool(table.commit == ctx.commit),
            ).count()
            row_counts[table_key] = int(cast("int", count_expr.execute()))
        except (RuntimeError, OSError, DuckDBCatalogException, IbisError):
            row_counts[table_key] = 0

    return row_counts


def compute_row_count(
    ctx: TargetExecutionContext,
    table_key: str,
) -> int:
    """Compute row count for a single table in the current snapshot.

    Convenience wrapper around ``compute_row_counts`` for single-table lookups.

    Parameters
    ----------
    ctx
        Execution context with gateway and snapshot.
    table_key
        Table key to count rows for.

    Returns
    -------
    int
        Number of rows in the table matching the current snapshot.
        Returns 0 if the table doesn't exist.
    """
    counts = compute_row_counts(ctx, [table_key])
    return counts.get(table_key, 0)
