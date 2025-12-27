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
from typing import TYPE_CHECKING

from codeintel.storage.duckdb_types import DuckDBRelation
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


def build_coverage_functions_expr(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> DuckDBRelation | None:
    """Build a DuckDB relation for coverage_functions without writing.

    Join function and method GOIDs with coverage line spans to compute
    per-function execution ratios. This is the pure compute version that
    returns a DuckDB relation for materialization via Hamilton.

    Parameters
    ----------
    gateway
        Gateway providing access to the DuckDB connection.
    snapshot
        Repository and commit identifiers that scope the aggregation.

    Returns
    -------
    DuckDBRelation | None
        DuckDB relation for analytics.coverage_functions, or None if
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
    >>> from codeintel.build.meta.contract_catalog import persist_contract_catalog_to_connection
    >>> def _seed(con):
    ...     persist_contract_catalog_to_connection(con, inputs={"source": "coverage_example"})
    >>> gateway = open_memory_gateway(seed_contract_catalog=_seed)
    >>> # ... setup tables ...
    >>> expr = build_coverage_functions_expr(gateway, snapshot)
    >>> # expr is a DuckDB relation ready for materialization
    """
    LOG.info(
        "Building coverage_functions expression for repo=%s commit=%s",
        snapshot.repo,
        snapshot.commit,
    )

    try:
        return gateway.con.sql(
            """
            WITH goids AS (
                SELECT
                    goid_h128,
                    urn,
                    repo,
                    commit,
                    rel_path,
                    language,
                    kind,
                    qualname,
                    start_line,
                    end_line
                FROM core.goids
                WHERE repo = $repo
                  AND commit = $commit
                  AND kind IN ('function', 'method')
            ),
            coverage AS (
                SELECT
                    repo,
                    commit,
                    rel_path,
                    line,
                    is_executable,
                    is_covered
                FROM analytics.coverage_lines
                WHERE repo = $repo
                  AND commit = $commit
            ),
            joined AS (
                SELECT
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
                    coverage.is_executable,
                    coverage.is_covered
                FROM goids
                LEFT JOIN coverage
                  ON goids.repo = coverage.repo
                 AND goids.commit = coverage.commit
                 AND goids.rel_path = coverage.rel_path
                 AND coverage.line >= goids.start_line
                 AND coverage.line <= COALESCE(goids.end_line, goids.start_line)
            ),
            aggregated AS (
                SELECT
                    goid_h128,
                    urn,
                    repo,
                    commit,
                    rel_path,
                    language,
                    kind,
                    qualname,
                    start_line,
                    end_line,
                    SUM(CASE WHEN is_executable THEN 1 ELSE 0 END) AS executable_lines_raw,
                    SUM(
                        CASE
                            WHEN is_executable AND is_covered THEN 1
                            ELSE 0
                        END
                    ) AS covered_lines_raw
                FROM joined
                GROUP BY
                    goid_h128,
                    urn,
                    repo,
                    commit,
                    rel_path,
                    language,
                    kind,
                    qualname,
                    start_line,
                    end_line
            )
            SELECT
                goid_h128 AS function_goid_h128,
                urn,
                repo,
                commit,
                rel_path,
                language,
                kind,
                qualname,
                start_line,
                end_line,
                COALESCE(executable_lines_raw, 0) AS executable_lines,
                COALESCE(covered_lines_raw, 0) AS covered_lines,
                CASE
                    WHEN COALESCE(executable_lines_raw, 0) = 0 THEN NULL
                    ELSE CAST(COALESCE(covered_lines_raw, 0) AS DOUBLE)
                         / NULLIF(CAST(COALESCE(executable_lines_raw, 0) AS DOUBLE), 0)
                END AS coverage_ratio,
                COALESCE(covered_lines_raw, 0) > 0 AS tested,
                CASE
                    WHEN COALESCE(executable_lines_raw, 0) = 0 THEN 'no_executable_code'
                    WHEN COALESCE(covered_lines_raw, 0) = 0 THEN 'no_tests'
                    ELSE ''
                END AS untested_reason,
                NOW() AS created_at
            FROM aggregated
            """,
            {"repo": snapshot.repo, "commit": snapshot.commit},
        )
    except DuckDBError as exc:
        LOG.warning("coverage_functions: failed to access tables: %s", exc)
        return None


__all__ = ["build_coverage_functions_expr"]
