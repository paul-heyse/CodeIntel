"""Subsystem cache row builders for profile and coverage caches."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsSubsystemCoverageCacheRow as SubsystemCoverageCacheRow,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsSubsystemProfileCacheRow as SubsystemProfileCacheRow,
)
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.query_results import iter_tuples_from_arrow_reader

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

PROFILE_CACHE_COLUMNS = (
    "repo",
    "commit",
    "subsystem_id",
    "name",
    "description",
    "module_count",
    "modules_json",
    "entrypoints_json",
    "internal_edge_count",
    "external_edge_count",
    "fan_in",
    "fan_out",
    "function_count",
    "avg_risk_score",
    "max_risk_score",
    "high_risk_function_count",
    "risk_level",
    "import_in_degree",
    "import_out_degree",
    "import_pagerank",
    "import_betweenness",
    "import_closeness",
    "import_layer",
    "created_at",
)

COVERAGE_CACHE_COLUMNS = (
    "repo",
    "commit",
    "subsystem_id",
    "name",
    "description",
    "module_count",
    "function_count",
    "risk_level",
    "avg_risk_score",
    "max_risk_score",
    "test_count",
    "passed_test_count",
    "failed_test_count",
    "skipped_test_count",
    "xfail_test_count",
    "flaky_test_count",
    "total_functions_covered",
    "avg_functions_covered",
    "max_functions_covered",
    "min_functions_covered",
    "function_coverage_ratio",
    "created_at",
)

PROFILE_CACHE_SQL = """
    SELECT
        s.repo,
        s.commit,
        s.subsystem_id,
        s.name,
        s.description,
        s.module_count,
        s.modules_json,
        s.entrypoints_json,
        s.internal_edge_count,
        s.external_edge_count,
        s.fan_in,
        s.fan_out,
        s.function_count,
        s.avg_risk_score,
        s.max_risk_score,
        s.high_risk_function_count,
        s.risk_level,
        gm.import_in_degree,
        gm.import_out_degree,
        gm.import_pagerank,
        gm.import_betweenness,
        gm.import_closeness,
        gm.import_layer,
        s.created_at
    FROM analytics.subsystems s
    LEFT JOIN analytics.subsystem_graph_metrics gm
      ON gm.repo = s.repo
     AND gm.commit = s.commit
     AND gm.subsystem_id = s.subsystem_id
    WHERE s.repo = ?
      AND s.commit = ?
"""

COVERAGE_CACHE_SQL = """
    WITH cov AS (
        SELECT
            repo,
            commit,
            primary_subsystem_id AS subsystem_id,
            COUNT(*) AS test_count,
            SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) AS passed_test_count,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_test_count,
            SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) AS skipped_test_count,
            SUM(CASE WHEN status = 'xfail' THEN 1 ELSE 0 END) AS xfail_test_count,
            SUM(CASE WHEN coalesce(flaky, FALSE) THEN 1 ELSE 0 END) AS flaky_test_count,
            SUM(coalesce(functions_covered_count, 0)) AS total_functions_covered,
            AVG(coalesce(functions_covered_count, 0)) AS avg_functions_covered,
            MAX(coalesce(functions_covered_count, 0)) AS max_functions_covered,
            MIN(coalesce(functions_covered_count, 0)) AS min_functions_covered
        FROM analytics.test_profile
        WHERE primary_subsystem_id IS NOT NULL
          AND repo = ?
          AND commit = ?
        GROUP BY repo, commit, primary_subsystem_id
    )
    SELECT
        s.repo,
        s.commit,
        s.subsystem_id,
        s.name,
        s.description,
        s.module_count,
        s.function_count,
        s.risk_level,
        s.avg_risk_score,
        s.max_risk_score,
        cov.test_count,
        cov.passed_test_count,
        cov.failed_test_count,
        cov.skipped_test_count,
        cov.xfail_test_count,
        cov.flaky_test_count,
        cov.total_functions_covered,
        cov.avg_functions_covered,
        cov.max_functions_covered,
        cov.min_functions_covered,
        CASE
            WHEN s.function_count = 0 THEN NULL
            ELSE cov.total_functions_covered * 1.0 / s.function_count
        END AS function_coverage_ratio,
        s.created_at
    FROM analytics.subsystems s
    LEFT JOIN cov
      ON cov.repo = s.repo
     AND cov.commit = s.commit
     AND cov.subsystem_id = s.subsystem_id
    WHERE s.repo = ?
      AND s.commit = ?
"""


def _rows_to_dicts(
    columns: tuple[str, ...],
    rows: Iterable[tuple[object, ...]],
) -> list[dict[str, object]]:
    return [dict(zip(columns, row, strict=True)) for row in rows]


def build_subsystem_profile_cache_rows(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> list[SubsystemProfileCacheRow]:
    """Build cache rows for analytics.subsystem_profile_cache.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.

    Returns
    -------
    list[SubsystemProfileCacheRow]
        Cache rows for subsystem profiles.
    """
    reader = gateway.execute(
        PROFILE_CACHE_SQL,
        [snapshot.repo, snapshot.commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    rows = iter_tuples_from_arrow_reader(reader)
    return [
        cast("SubsystemProfileCacheRow", row) for row in _rows_to_dicts(PROFILE_CACHE_COLUMNS, rows)
    ]


def build_subsystem_coverage_cache_rows(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> list[SubsystemCoverageCacheRow]:
    """Build cache rows for analytics.subsystem_coverage_cache.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.

    Returns
    -------
    list[SubsystemCoverageCacheRow]
        Cache rows for subsystem coverage.
    """
    reader = gateway.execute(
        COVERAGE_CACHE_SQL,
        [
            snapshot.repo,
            snapshot.commit,
            snapshot.repo,
            snapshot.commit,
        ],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    rows = iter_tuples_from_arrow_reader(reader)
    return [
        cast("SubsystemCoverageCacheRow", row)
        for row in _rows_to_dicts(COVERAGE_CACHE_COLUMNS, rows)
    ]


__all__ = [
    "build_subsystem_coverage_cache_rows",
    "build_subsystem_profile_cache_rows",
]
