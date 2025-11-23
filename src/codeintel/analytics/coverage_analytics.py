"""
Aggregate line-level coverage data into function-level coverage statistics.

The utilities here join GOIDs with coverage line spans to compute per-function
execution ratios, which downstream risk scoring relies on.
"""

from __future__ import annotations

import logging

import duckdb

from codeintel.config.models import CoverageAnalyticsConfig
from codeintel.config.schemas.sql_builder import ensure_schema

log = logging.getLogger(__name__)


def compute_coverage_functions(
    con: duckdb.DuckDBPyConnection,
    cfg: CoverageAnalyticsConfig,
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
    con : duckdb.DuckDBPyConnection
        Connection with `core.goids` and `analytics.coverage_lines` populated.
    cfg : CoverageAnalyticsConfig
        Repository and commit identifiers that scope the aggregation.

    Notes
    -----
    - Existing rows for the same repo/commit are deleted before insertion,
      making the operation idempotent for a given snapshot.
    - Time complexity is proportional to the number of functions and covered
      lines for the specified commit.

    Examples
    --------
    >>> import duckdb
    >>> con = duckdb.connect(":memory:")
    >>> con.execute("CREATE SCHEMA core")
    <duckdb.DuckDBPyConnection object ...>
    >>> con.execute("CREATE SCHEMA analytics")
    <duckdb.DuckDBPyConnection object ...>
    >>> con.execute(
    ...     "CREATE TABLE core.goids(urn VARCHAR, repo VARCHAR, commit VARCHAR,"
    ...     " rel_path VARCHAR, language VARCHAR, kind VARCHAR, qualname VARCHAR,"
    ...     " goid_h128 VARCHAR, start_line INTEGER, end_line INTEGER)"
    ... )
    <duckdb.DuckDBPyConnection object ...>
    >>> con.execute(
    ...     "CREATE TABLE analytics.coverage_lines(repo VARCHAR, commit VARCHAR,"
    ...     " rel_path VARCHAR, line INTEGER, is_executable BOOLEAN, is_covered BOOLEAN)"
    ... )
    <duckdb.DuckDBPyConnection object ...>
    >>> con.execute(
    ...     "CREATE TABLE analytics.coverage_functions("
    ...     "function_goid_h128 VARCHAR, urn VARCHAR, repo VARCHAR, commit VARCHAR,"
    ...     " rel_path VARCHAR, language VARCHAR, kind VARCHAR, qualname VARCHAR,"
    ...     " start_line INTEGER, end_line INTEGER, executable_lines INTEGER,"
    ...     " covered_lines INTEGER, coverage_ratio DOUBLE, tested BOOLEAN,"
    ...     " untested_reason VARCHAR, created_at TIMESTAMP)"
    ... )
    <duckdb.DuckDBPyConnection object ...>
    >>> con.execute(
    ...     "INSERT INTO core.goids VALUES ("
    ...     " 'urn:func', 'demo', 'abc', 'foo.py', 'python', 'function',"
    ...     " 'foo', 'h128', 1, 3)"
    ... )
    <duckdb.DuckDBPyConnection object ...>
    >>> con.execute(
    ...     "INSERT INTO analytics.coverage_lines VALUES "
    ...     " ('demo', 'abc', 'foo.py', 1, TRUE, TRUE),"
    ...     " ('demo', 'abc', 'foo.py', 2, TRUE, FALSE),"
    ...     " ('demo', 'abc', 'foo.py', 3, FALSE, FALSE)"
    ... )
    <duckdb.DuckDBPyConnection object ...>
    >>> cfg = CoverageAnalyticsConfig(repo="demo", commit="abc")
    >>> compute_coverage_functions(con, cfg)
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

    ensure_schema(con, "analytics.coverage_lines")
    ensure_schema(con, "analytics.coverage_functions")

    # Clear any previous rows for this repo/commit.
    con.execute(
        """
        DELETE FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    )

    # Insert aggregated coverage per function.
    #
    # Notes:
    # - We restrict to function/method GOIDs only.
    # - We join coverage_lines by (repo, commit, rel_path) + line span.
    # - Lines with no coverage_lines rows are treated as non-executable.
    # - coverage_ratio is NULL when there are no executable lines. :contentReference[oaicite:4]{index=4}
    insert_sql = """
        INSERT INTO analytics.coverage_functions (
            function_goid_h128,
            urn,
            repo,
            commit,
            rel_path,
            language,
            kind,
            qualname,
            start_line,
            end_line,
            executable_lines,
            covered_lines,
            coverage_ratio,
            tested,
            untested_reason,
            created_at
        )
        SELECT
            g.goid_h128                    AS function_goid_h128,
            g.urn,
            g.repo,
            g.commit,
            g.rel_path,
            g.language,
            g.kind,
            g.qualname,
            g.start_line,
            g.end_line,
            COUNT(*) FILTER (WHERE c.is_executable) AS executable_lines,
            COUNT(*) FILTER (WHERE c.is_executable AND c.is_covered) AS covered_lines,
            CASE
                WHEN COUNT(*) FILTER (WHERE c.is_executable) = 0 THEN NULL
                ELSE
                    CAST(
                        COUNT(*) FILTER (WHERE c.is_executable AND c.is_covered)
                        AS DOUBLE
                    )
                    / COUNT(*) FILTER (WHERE c.is_executable)
            END AS coverage_ratio,
            COUNT(*) FILTER (WHERE c.is_executable AND c.is_covered) > 0 AS tested,
            CASE
                WHEN COUNT(*) FILTER (WHERE c.is_executable) = 0 THEN 'no_executable_code'
                WHEN COUNT(*) FILTER (WHERE c.is_executable AND c.is_covered) = 0 THEN 'no_tests'
                ELSE ''
            END AS untested_reason,
            NOW() AS created_at
        FROM core.goids g
        LEFT JOIN analytics.coverage_lines c
          ON c.repo = g.repo
         AND c.commit = g.commit
         AND c.rel_path = g.rel_path
         AND c.line BETWEEN g.start_line AND COALESCE(g.end_line, g.start_line)
        WHERE g.repo = ?
          AND g.commit = ?
          AND g.kind IN ('function', 'method')
        GROUP BY
            g.goid_h128,
            g.urn,
            g.repo,
            g.commit,
            g.rel_path,
            g.language,
            g.kind,
            g.qualname,
            g.start_line,
            g.end_line;
    """

    con.execute(insert_sql, [cfg.repo, cfg.commit])

    row = con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchone()
    n = int(row[0]) if row is not None else 0

    log.info("coverage_functions populated: %d rows for %s@%s", n, cfg.repo, cfg.commit)
