"""
Aggregate line-level coverage data into function-level coverage statistics.

The utilities here join GOIDs with coverage line spans to compute per-function
execution ratios, which downstream risk scoring relies on.
"""

from __future__ import annotations

import logging
from typing import cast

import ibis
import ibis.expr.types as it

from codeintel.analytics.compute.ibis_utils import (
    bool_and,
    literal_sequence,
    safe_ratio,
    zero_if_null,
)
from codeintel.config import CoverageAnalyticsStepConfig
from codeintel.storage.gateway import DuckDBError, StorageGateway
from codeintel.storage.ibis_types import gt, ibis_bool

log = logging.getLogger(__name__)


def compute_coverage_functions(
    gateway: StorageGateway,
    cfg: CoverageAnalyticsStepConfig,
) -> None:
    """
    Populate `analytics.coverage_functions` by aggregating line coverage per GOID.

    Extended Summary
    ----------------
    The query joins function and method GOIDs with `analytics.coverage_lines`
    records to compute executable line counts, covered lines, and derived ratios.
    Results mirror the `coverage_functions` schema from `README_METADATA` so
    downstream risk scoring and reporting can reason about test completeness.

    Parameters
    ----------
    gateway : StorageGateway
        Gateway providing access to the DuckDB connection.
    cfg : CoverageAnalyticsStepConfig
        Repository and commit identifiers that scope the aggregation.

    Notes
    -----
    - Existing rows for the same repo/commit are deleted before insertion,
      making the operation idempotent for a given snapshot.
    - Time complexity is proportional to the number of functions and covered
      lines for the specified commit.

    Examples
    --------
    >>> from codeintel.storage.gateway import open_memory_gateway
    >>> gateway = open_memory_gateway()
    >>> con = gateway.con
    >>> _ = con.execute("CREATE SCHEMA core")
    >>> _ = con.execute("CREATE SCHEMA analytics")
    >>> _ = con.execute(
    ...     "CREATE TABLE core.goids(urn VARCHAR, repo VARCHAR, commit VARCHAR,"
    ...     " rel_path VARCHAR, language VARCHAR, kind VARCHAR, qualname VARCHAR,"
    ...     " goid_h128 VARCHAR, start_line INTEGER, end_line INTEGER)"
    ... )
    >>> _ = con.execute(
    ...     "CREATE TABLE analytics.coverage_lines(repo VARCHAR, commit VARCHAR,"
    ...     " rel_path VARCHAR, line INTEGER, is_executable BOOLEAN, is_covered BOOLEAN)"
    ... )
    >>> _ = con.execute(
    ...     "CREATE TABLE analytics.coverage_functions("
    ...     "function_goid_h128 VARCHAR, urn VARCHAR, repo VARCHAR, commit VARCHAR,"
    ...     " rel_path VARCHAR, language VARCHAR, kind VARCHAR, qualname VARCHAR,"
    ...     " start_line INTEGER, end_line INTEGER, executable_lines INTEGER,"
    ...     " covered_lines INTEGER, coverage_ratio DOUBLE, tested BOOLEAN,"
    ...     " untested_reason VARCHAR, created_at TIMESTAMP)"
    ... )
    >>> _ = con.execute(
    ...     "INSERT INTO core.goids VALUES ("
    ...     " 'urn:func', 'demo', 'abc', 'foo.py', 'python', 'function',"
    ...     " 'foo', 'h128', 1, 3)"
    ... )
    >>> _ = con.execute(
    ...     "INSERT INTO analytics.coverage_lines VALUES "
    ...     " ('demo', 'abc', 'foo.py', 1, TRUE, TRUE),"
    ...     " ('demo', 'abc', 'foo.py', 2, TRUE, FALSE),"
    ...     " ('demo', 'abc', 'foo.py', 3, FALSE, FALSE)"
    ... )
    >>> from codeintel.config import ConfigBuilder, SnapshotInit
    >>> cfg = ConfigBuilder.from_snapshot(
    ...     snapshot=SnapshotInit(repo="demo", commit="abc", repo_root=Path(".")),
    ... ).coverage_analytics()
    >>> compute_coverage_functions(gateway, cfg)
    >>> con.execute(
    ...     "SELECT executable_lines, covered_lines, coverage_ratio, tested"
    ...     " FROM analytics.coverage_functions"
    ... ).fetchall()
    [(2, 1, 0.5, True)]
    """
    log.info(
        "Computing coverage_functions for repo=%s commit=%s",
        cfg.repo,
        cfg.commit,
    )
    try:
        goids = gateway.ibis.table("core.goids")
        coverage = gateway.ibis.table("analytics.coverage_lines")
    except DuckDBError as exc:
        log.warning("coverage_functions: failed to access tables: %s", exc)
        return

    goids_filtered = _filter_goids(goids, cfg)
    coverage_filtered = _filter_coverage_lines(coverage, cfg)
    joined = _join_goids_with_coverage(goids_filtered, coverage_filtered)
    aggregated = _aggregate_coverage(joined, goids_filtered)
    result_expr = _enrich_coverage_results(aggregated)

    try:
        _write_coverage_results(gateway, cfg, result_expr)
    except DuckDBError as exc:
        log.warning("coverage_functions: failed to write results: %s", exc)
        return

    row_count_expr = result_expr.count()
    row_count = cast("int", row_count_expr.execute())
    log.info(
        "coverage_functions populated: %d rows for %s@%s",
        row_count,
        cfg.repo,
        cfg.commit,
    )


def _filter_goids(table: it.Table, cfg: CoverageAnalyticsStepConfig) -> it.Table:
    predicate = bool_and(
        ibis_bool(table.repo == cfg.repo),
        ibis_bool(table.commit == cfg.commit),
        ibis_bool(table.kind.isin(literal_sequence(["function", "method"]))),
    )
    return table.filter(predicate)


def _filter_coverage_lines(table: it.Table, cfg: CoverageAnalyticsStepConfig) -> it.Table:
    predicate = bool_and(
        ibis_bool(table.repo == cfg.repo),
        ibis_bool(table.commit == cfg.commit),
    )
    return table.filter(predicate)


def _join_goids_with_coverage(goids: it.Table, coverage: it.Table) -> it.Table:
    end_line = ibis.coalesce(goids.end_line, goids.start_line)
    join_predicates = [
        ibis_bool(goids.repo == coverage.repo),
        ibis_bool(goids.commit == coverage.commit),
        ibis_bool(goids.rel_path == coverage.rel_path),
        ibis_bool(coverage.line >= goids.start_line),
        ibis_bool(coverage.line <= end_line),
    ]
    return goids.left_join(coverage, join_predicates)


def _aggregate_coverage(joined: it.Table, goids: it.Table) -> it.Table:
    executable = ibis_bool(joined["is_executable"])
    covered = ibis_bool(joined["is_covered"])
    executable_int = cast("it.IntegerColumn", executable.cast("int64"))
    covered_int = cast("it.IntegerColumn", bool_and(executable, covered).cast("int64"))
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


def _enrich_coverage_results(aggregated: it.Table) -> it.Table:
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
        (
            ibis.case()
            .when(no_executable_code, "no_executable_code")
            .when(no_tests, "no_tests")
            .else_("")
            .end()
        ).name("untested_reason"),
        ibis.now().name("created_at"),
    )


def _write_coverage_results(
    gateway: StorageGateway,
    cfg: CoverageAnalyticsStepConfig,
    result_expr: it.Table,
) -> None:
    table = gateway.ibis.table("analytics.coverage_functions")
    where = bool_and(
        ibis_bool(table.repo == cfg.repo),
        ibis_bool(table.commit == cfg.commit),
    )
    gateway.ibis.delete("analytics.coverage_functions", where=where)
    gateway.ibis.write("analytics.coverage_functions", result_expr)
