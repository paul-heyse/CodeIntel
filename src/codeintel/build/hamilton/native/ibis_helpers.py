"""Ibis helper utilities for native Hamilton targets.

This module provides utilities for common Ibis operations in native targets,
reducing boilerplate and ensuring consistent snapshot filtering.

Example
-------
>>> filtered = filter_for_snapshot(modules_table, env.snapshot)
>>> table_dict = filter_tables_for_snapshot(
...     env.snapshot,
...     modules=q__core__modules,
...     function_metrics=q__analytics__function_metrics,
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from codeintel.storage.ibis_types import and_predicates

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from codeintel.config.primitives import SnapshotRef


def filter_for_snapshot(table: ir.Table, snapshot: SnapshotRef) -> ir.Table:
    """Filter an Ibis table to the current snapshot.

    This utility replaces the common pattern of filtering by repo and commit
    that appears throughout native Hamilton targets.

    Parameters
    ----------
    table
        Ibis table expression with repo and commit columns.
    snapshot
        Snapshot reference providing repo and commit values.

    Returns
    -------
    ir.Table
        Filtered table expression.

    Examples
    --------
    >>> filtered = filter_for_snapshot(q__core__modules, env.snapshot)
    >>> # Equivalent to:
    >>> # q__core__modules.filter(and_predicates(
    >>> #     q__core__modules.repo == env.snapshot.repo,
    >>> #     q__core__modules.commit == env.snapshot.commit,
    >>> # ))
    """
    return table.filter(
        cast(
            "Any",
            and_predicates(
                table.repo == snapshot.repo,
                table.commit == snapshot.commit,
            ),
        )
    )


def filter_tables_for_snapshot(
    snapshot: SnapshotRef,
    **tables: ir.Table,
) -> dict[str, ir.Table]:
    """Filter multiple tables to the current snapshot.

    This utility handles the common pattern of filtering many input tables
    by the same snapshot in a single call.

    Parameters
    ----------
    snapshot
        Snapshot reference providing repo and commit values.
    **tables
        Named Ibis table expressions to filter.

    Returns
    -------
    dict[str, ir.Table]
        Dictionary of filtered tables with same keys.

    Examples
    --------
    >>> tables = filter_tables_for_snapshot(
    ...     env.snapshot,
    ...     modules=q__core__modules,
    ...     metrics=q__analytics__function_metrics,
    ... )
    >>> # Access filtered tables
    >>> modules_filtered = tables["modules"]
    >>> metrics_filtered = tables["metrics"]
    """
    return {name: filter_for_snapshot(table, snapshot) for name, table in tables.items()}


def select_snapshot_columns(table: ir.Table, *columns: str) -> ir.Table:
    """Select specified columns plus repo and commit from a table.

    This utility ensures repo and commit are always included when selecting
    columns from a table, which is required for snapshot-aware operations.

    Parameters
    ----------
    table
        Ibis table expression with repo and commit columns.
    *columns
        Column names to select (repo and commit are added automatically).

    Returns
    -------
    ir.Table
        Table expression with specified columns plus repo and commit.

    Examples
    --------
    >>> selected = select_snapshot_columns(
    ...     q__analytics__function_metrics,
    ...     "function_goid_h128",
    ...     "cyclomatic_complexity",
    ... )
    >>> # Results in columns: function_goid_h128, cyclomatic_complexity, repo, commit
    """
    # Build column set ensuring repo and commit are included
    column_set = {"repo", "commit", *columns}
    return cast("Any", table.select(*column_set))


__all__ = [
    "filter_for_snapshot",
    "filter_tables_for_snapshot",
    "select_snapshot_columns",
]
