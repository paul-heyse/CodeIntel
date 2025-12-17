"""Pure compute functions for coverage_functions analytics.

This module provides pure compute functions that build Ibis expressions for
coverage analytics without performing writes. Use these functions with
Hamilton materializers for persistence.

Example
-------
>>> from codeintel.analytics.compute.coverage import build_coverage_functions_expr
>>> expr = build_coverage_functions_expr(gateway, snapshot)
>>> if expr is not None:
...     ref = materialize_table(ctx, "analytics.coverage_functions", expr)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import ibis

from codeintel.analytics.compute.ibis_utils import (
    literal_sequence,
    safe_ratio,
    zero_if_null,
)
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import and_predicates, gt, ibis_bool

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


def build_coverage_functions_expr(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> ir.Table | None:
    """Build Ibis expression for coverage_functions without writing.

    Join function and method GOIDs with coverage line spans to compute
    per-function execution ratios. This is the pure compute version that
    returns an Ibis expression for materialization via Hamilton.

    Parameters
    ----------
    gateway
        Gateway providing access to the DuckDB connection.
    snapshot
        Repository and commit identifiers that scope the aggregation.

    Returns
    -------
    ir.Table | None
        Ibis table expression for analytics.coverage_functions, or None if
        the required source tables cannot be accessed.

    Notes
    -----
    The expression computes:
    - executable_lines: Count of lines marked as executable
    - covered_lines: Count of lines both executable and covered
    - coverage_ratio: covered_lines / executable_lines
    - tested: True if covered_lines > 0
    - untested_reason: "no_executable_code" or "no_tests" or ""

    Examples
    --------
    >>> from codeintel.storage.gateway import open_memory_gateway
    >>> gateway = open_memory_gateway()
    >>> # ... setup tables ...
    >>> expr = build_coverage_functions_expr(gateway, snapshot)
    >>> # expr is an Ibis Table expression ready for materialization
    """
    LOG.info(
        "Building coverage_functions expression for repo=%s commit=%s",
        snapshot.repo,
        snapshot.commit,
    )

    try:
        goids = gateway.ibis.table("core.goids")
        coverage = gateway.ibis.table("analytics.coverage_lines")
    except DuckDBError as exc:
        LOG.warning("coverage_functions: failed to access tables: %s", exc)
        return None

    return build_coverage_functions_expr_from_tables(goids, coverage, snapshot=snapshot)


def build_coverage_functions_expr_from_tables(
    goids: ir.Table,
    coverage: ir.Table,
    *,
    snapshot: SnapshotRef,
) -> ir.Table:
    """Build Ibis expression for coverage_functions from pre-loaded tables.

    Parameters
    ----------
    goids
        Ibis table expression for ``core.goids``.
    coverage
        Ibis table expression for ``analytics.coverage_lines``.
    snapshot
        Repository and commit identifiers that scope the aggregation.

    Returns
    -------
    ir.Table
        Ibis table expression for ``analytics.coverage_functions``.
    """
    goids_filtered = filter_goids_for_snapshot(goids, snapshot)
    coverage_filtered = filter_coverage_lines_for_snapshot(coverage, snapshot)
    joined = join_goids_with_coverage_lines(goids_filtered, coverage_filtered)
    aggregated = aggregate_coverage_lines(joined, goids_filtered)
    return enrich_coverage_results(aggregated)


def filter_goids_for_snapshot(table: ir.Table, snapshot: SnapshotRef) -> ir.Table:
    """Filter GOIDs to functions/methods for the given snapshot.

    Parameters
    ----------
    table
        Ibis table for core.goids.
    snapshot
        Snapshot reference for filtering.

    Returns
    -------
    ir.Table
        Filtered table with only function/method GOIDs for the snapshot.
    """
    predicate = and_predicates(
        ibis_bool(table.repo == snapshot.repo),
        ibis_bool(table.commit == snapshot.commit),
        ibis_bool(table.kind.isin(literal_sequence(["function", "method"]))),
    )
    return table.filter(predicate)


def filter_coverage_lines_for_snapshot(table: ir.Table, snapshot: SnapshotRef) -> ir.Table:
    """Filter coverage lines for the given snapshot.

    Parameters
    ----------
    table
        Ibis table for analytics.coverage_lines.
    snapshot
        Snapshot reference for filtering.

    Returns
    -------
    ir.Table
        Filtered table with coverage lines for the snapshot.
    """
    predicate = and_predicates(
        ibis_bool(table.repo == snapshot.repo),
        ibis_bool(table.commit == snapshot.commit),
    )
    return table.filter(predicate)


def join_goids_with_coverage_lines(goids: ir.Table, coverage: ir.Table) -> ir.Table:
    """Join GOIDs with coverage lines based on file path and line ranges.

    Parameters
    ----------
    goids
        Filtered GOIDs table.
    coverage
        Filtered coverage lines table.

    Returns
    -------
    ir.Table
        Joined table with GOIDs and their coverage line data.

    Notes
    -----
    The join matches coverage lines where:
    - Same repo, commit, and rel_path
    - Line number is between function start_line and end_line
    """
    end_line = ibis.coalesce(goids.end_line, goids.start_line)
    join_predicates = [
        ibis_bool(goids.repo == coverage.repo),
        ibis_bool(goids.commit == coverage.commit),
        ibis_bool(goids.rel_path == coverage.rel_path),
        ibis_bool(coverage.line >= goids.start_line),
        ibis_bool(coverage.line <= end_line),
    ]
    return goids.left_join(coverage, join_predicates)


def aggregate_coverage_lines(joined: ir.Table, goids: ir.Table) -> ir.Table:
    """Aggregate coverage metrics per function.

    Parameters
    ----------
    joined
        Joined GOIDs and coverage lines table.
    goids
        Original filtered GOIDs table (for column references).

    Returns
    -------
    ir.Table
        Aggregated table with executable_lines_raw and covered_lines_raw.
    """
    executable = ibis_bool(joined["is_executable"])
    covered = ibis_bool(joined["is_covered"])
    executable_int = cast("ir.IntegerColumn", executable.cast("int64"))
    covered_int = cast("ir.IntegerColumn", and_predicates(executable, covered).cast("int64"))
    grouped = joined.group_by(
        goids.goid_h128,
        goids.urn,
        goids.repo,
        goids.commit,
        goids.rel_path,
        goids.language,
        goids.kind,
        goids.qualname,
        goids.start_line,
        goids.end_line,
    )
    return grouped.aggregate(
        executable_lines_raw=executable_int.sum(),
        covered_lines_raw=covered_int.sum(),
    )


def enrich_coverage_results(aggregated: ir.Table) -> ir.Table:
    """Enrich aggregated coverage with derived metrics.

    Parameters
    ----------
    aggregated
        Aggregated coverage table with raw counts.

    Returns
    -------
    ir.Table
        Final table with all coverage_functions columns including:
        coverage_ratio, tested flag, untested_reason, and created_at.
    """
    exec_lines = zero_if_null(aggregated.executable_lines_raw)
    covered_lines = zero_if_null(aggregated.covered_lines_raw)
    coverage_ratio = safe_ratio(covered_lines, exec_lines)
    no_executable_code = ibis_bool(exec_lines == 0)
    no_tests = ibis_bool(covered_lines == 0)

    return aggregated.select(
        aggregated.goid_h128.name("function_goid_h128"),
        aggregated.urn,
        aggregated.repo,
        aggregated.commit,
        aggregated.rel_path,
        aggregated.language,
        aggregated.kind,
        aggregated.qualname,
        aggregated.start_line,
        aggregated.end_line,
        exec_lines.name("executable_lines"),
        covered_lines.name("covered_lines"),
        coverage_ratio.name("coverage_ratio"),
        gt(covered_lines, 0).name("tested"),
        ibis.cases(
            (no_executable_code, "no_executable_code"),
            (no_tests, "no_tests"),
            else_="",
        ).name("untested_reason"),
        ibis.now().name("created_at"),
    )


__all__ = [
    "aggregate_coverage_lines",
    "build_coverage_functions_expr",
    "build_coverage_functions_expr_from_tables",
    "enrich_coverage_results",
    "filter_coverage_lines_for_snapshot",
    "filter_goids_for_snapshot",
    "join_goids_with_coverage_lines",
]
