"""Shared helper utilities for build plugins.

Provides common functionality used across multiple plugin implementations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway

__all__ = [
    "compute_row_count",
    "compute_row_counts",
    "filter_paths",
    "get_source_root",
    "is_test_path",
]

log = logging.getLogger(__name__)


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

    >>> counts = compute_row_counts(ctx, ["analytics.function_metrics"])
    >>> print(counts["analytics.function_metrics"])
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


def get_source_root(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    fallback: Path | None = None,
) -> Path:
    """Retrieve source root from core.snapshots with fallback.

    Look up the source root for the given repository snapshot from the
    core.snapshots table. Returns a fallback path if not found.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo
        Repository identifier.
    commit
        Commit SHA.
    fallback
        Fallback path if not found. Defaults to ``Path.cwd()``.

    Returns
    -------
    Path
        Absolute path to the source root.
    """
    from codeintel.storage.gateway import DuckDBError  # noqa: PLC0415

    try:
        snapshots = gateway.ibis.table("core.snapshots")
        repo_filter = cast("Any", snapshots.repo == repo)
        commit_filter = cast("Any", snapshots.commit == commit)
        expr = snapshots.filter(repo_filter & commit_filter).select(snapshots.source_root).limit(1)
        df = expr.execute()
        if not getattr(df, "empty", True):
            value = df.iloc[0][0]
            if value:
                return Path(str(value))
    except DuckDBError as exc:
        log.debug("get_source_root: Could not get source root: %s", exc)
    return fallback or Path.cwd()


def is_test_path(path: str) -> bool:
    """Check whether a path appears to be a test file.

    Uses common Python test file naming conventions to detect test files:
    - Files in a ``tests/`` directory
    - Files ending in ``_test.py``
    - Files containing ``/test_`` in the path
    - Files starting with ``test_``

    Parameters
    ----------
    path
        Relative file path to check.

    Returns
    -------
    bool
        True if the path matches test file patterns.

    Examples
    --------
    >>> is_test_path("tests/test_module.py")
    True
    >>> is_test_path("src/module.py")
    False
    >>> is_test_path("test_utils.py")
    True
    """
    lowered = path.lower()
    return (
        "tests/" in lowered
        or lowered.endswith("_test.py")
        or "/test_" in lowered
        or lowered.startswith("test_")
    )


def filter_paths(
    paths: Iterable[str],
    *,
    scope_paths: list[str] | None = None,
    include_tests: bool = True,
) -> list[str]:
    """Filter paths by scope and test inclusion.

    Provides a unified path filtering mechanism for plugins that need to
    restrict processing to specific directories and optionally exclude test files.

    Parameters
    ----------
    paths
        Paths to filter.
    scope_paths
        Optional list of path prefixes to include. If None or empty,
        all paths are included.
    include_tests
        Whether to include test files. Uses ``is_test_path()`` for detection.
        Defaults to True.

    Returns
    -------
    list[str]
        Filtered list of paths.

    Examples
    --------
    Filter to a specific directory:

    >>> filter_paths(["src/a.py", "lib/b.py"], scope_paths=["src/"])
    ['src/a.py']

    Exclude test files:

    >>> filter_paths(["src/main.py", "tests/test_main.py"], include_tests=False)
    ['src/main.py']
    """
    result = list(paths)

    if scope_paths:
        prefixes = tuple(scope_paths)
        result = [path for path in result if path.startswith(prefixes)]

    if not include_tests:
        result = [path for path in result if not is_test_path(path)]

    return result
