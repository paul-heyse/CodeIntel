# src/codeintel/analytics/coverage_analytics.py

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb

log = logging.getLogger(__name__)


@dataclass
class CoverageAnalyticsConfig:
    """
    Configuration for computing per-function coverage.

    We aggregate coverage_lines over GOID spans for functions/methods,
    matching the coverage_functions spec from README_METADATA. :contentReference[oaicite:2]{index=2}
    """
    repo: str
    commit: str


def compute_coverage_functions(
    con: duckdb.DuckDBPyConnection,
    cfg: CoverageAnalyticsConfig,
) -> None:
    """
    Populate analytics.coverage_functions by joining:

      - core.goids (functions + methods)
      - analytics.coverage_lines (line-level coverage) :contentReference[oaicite:3]{index=3}

    The resulting table has columns:

      function_goid_h128, urn, repo, commit, rel_path, language, kind,
      qualname, start_line, end_line, executable_lines, covered_lines,
      coverage_ratio, tested, untested_reason, created_at.
    """
    log.info(
        "Computing coverage_functions for repo=%s commit=%s",
        cfg.repo,
        cfg.commit,
    )

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

    n = con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchone()[0]

    log.info("coverage_functions populated: %d rows for %s@%s", n, cfg.repo, cfg.commit)
