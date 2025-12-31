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

from codeintel.analytics.duckdb_helpers import aggregate_relation
from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression, DuckDBRelation
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
        predicate = (ColumnExpression("repo") == ConstantExpression(snapshot.repo)) & (
            ColumnExpression("commit") == ConstantExpression(snapshot.commit)
        )
        kind_literals = [ConstantExpression("function"), ConstantExpression("method")]
        goids = (
            gateway.relation_from_table_key("core.goids")
            .filter(predicate)
            .filter(ColumnExpression("kind").isin(*kind_literals))
            .select(
                "goid_h128",
                "urn",
                "repo",
                "commit",
                "rel_path",
                "language",
                "kind",
                "qualname",
                "start_line",
                "end_line",
            )
            .set_alias("goids")
        )
        coverage = (
            gateway.relation_from_table_key("analytics.coverage_lines")
            .filter(predicate)
            .select("repo", "commit", "rel_path", "line", "is_executable", "is_covered")
            .set_alias("coverage")
        )
        joined = goids.join(
            coverage,
            "goids.repo = coverage.repo "
            "AND goids.commit = coverage.commit "
            "AND goids.rel_path = coverage.rel_path "
            "AND coverage.line >= goids.start_line "
            "AND coverage.line <= coalesce(goids.end_line, goids.start_line)",
            how="left",
        )
        aggregated = aggregate_relation(
            joined,
            aggs=[
                "sum(case when is_executable then 1 else 0 end) as executable_lines_raw",
                "sum(case when is_executable and is_covered then 1 else 0 end) as covered_lines_raw",
            ],
            group_by=(
                "goid_h128, urn, repo, commit, rel_path, language, kind, qualname, "
                "start_line, end_line"
            ),
        )
        return aggregated.select(
            "goid_h128 as function_goid_h128",
            "urn",
            "repo",
            "commit",
            "rel_path",
            "language",
            "kind",
            "qualname",
            "start_line",
            "end_line",
            "coalesce(executable_lines_raw, 0) as executable_lines",
            "coalesce(covered_lines_raw, 0) as covered_lines",
            "case "
            "when coalesce(executable_lines_raw, 0) = 0 then null "
            "else cast(coalesce(covered_lines_raw, 0) as double) / "
            "nullif(cast(coalesce(executable_lines_raw, 0) as double), 0) "
            "end as coverage_ratio",
            "coalesce(covered_lines_raw, 0) > 0 as tested",
            "case "
            "when coalesce(executable_lines_raw, 0) = 0 then 'no_executable_code' "
            "when coalesce(covered_lines_raw, 0) = 0 then 'no_tests' "
            "else '' "
            "end as untested_reason",
            "now() as created_at",
        )
    except DuckDBError as exc:
        LOG.warning("coverage_functions: failed to access tables: %s", exc)
        return None


__all__ = ["build_coverage_functions_expr"]
