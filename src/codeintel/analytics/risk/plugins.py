"""Analytics plugin for risk factors aggregation."""

from __future__ import annotations

from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.analytics.subsystems import refresh_subsystem_caches
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.storage.gateway import StorageGateway


def _seed_catalog_modules(
    gateway: StorageGateway,
    catalog: FunctionCatalogProvider | None,
    *,
    repo: str,
    commit: str,
) -> bool:
    """
    Create a temporary table of modules from a catalog when available.

    Returns
    -------
    bool
        True when a temp table was created.
    """
    if catalog is None or not hasattr(catalog, "catalog"):
        return False
    module_by_path = catalog.catalog().module_by_path
    if not module_by_path:
        return False
    con = gateway.con
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE temp.catalog_modules (
            path VARCHAR,
            module VARCHAR,
            repo VARCHAR,
            commit VARCHAR,
            tags JSON,
            owners JSON
        )
        """
    )
    con.executemany(
        "INSERT INTO temp.catalog_modules VALUES (?, ?, ?, ?, ?, ?)",
        [(path, module, repo, commit, "[]", "[]") for path, module in module_by_path.items()],
    )
    return True


def _risk_factors_run(ctx: AnalyticsExecutionContext) -> object | None:
    """
    Reuse legacy risk factor aggregation inside the plugin harness.

    Returns
    -------
    None
        Results are persisted directly to storage.
    """
    gateway = ctx.gateway
    con = gateway.con
    catalog = ctx.catalog_provider

    con.execute(
        "DELETE FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    )

    use_catalog_modules = _seed_catalog_modules(gateway, catalog, repo=ctx.repo, commit=ctx.commit)

    risk_sql = """
        INSERT INTO analytics.goid_risk_factors
        SELECT
            fm.function_goid_h128,
            fm.urn,
            fm.repo,
            fm.commit,
            fm.rel_path,
            fm.language,
            fm.kind,
            fm.qualname,
            fm.loc,
            fm.logical_loc,
            fm.cyclomatic_complexity,
            fm.complexity_bucket,
            ft.typedness_bucket,
            ft.typedness_source,
            h.score                       AS hotspot_score,
            CAST(ty.annotation_ratio->>'params' AS DOUBLE) AS file_typed_ratio,
            sd.total_errors               AS static_error_count,
            sd.has_errors                 AS has_static_errors,
            cf.executable_lines,
            cf.covered_lines,
            cf.coverage_ratio,
            cf.tested,
            COALESCE(t_stats.test_count, 0)         AS test_count,
            COALESCE(t_stats.failing_test_count, 0) AS failing_test_count,
            COALESCE(t_stats.last_test_status, 'unknown') AS last_test_status,
            (
                COALESCE(1.0 - cf.coverage_ratio, 1.0) * 0.4 +
                CASE fm.complexity_bucket
                    WHEN 'high' THEN 0.4
                    WHEN 'medium' THEN 0.2
                    ELSE 0.0
                END +
                CASE WHEN sd.has_errors THEN 0.2 ELSE 0.0 END +
                CASE WHEN h.score > 0 THEN 0.1 ELSE 0.0 END
            ) AS risk_score,
            CASE
                WHEN (
                    COALESCE(1.0 - cf.coverage_ratio, 1.0) * 0.4 +
                    CASE fm.complexity_bucket
                        WHEN 'high' THEN 0.4
                        WHEN 'medium' THEN 0.2
                        ELSE 0.0
                    END +
                    CASE WHEN sd.has_errors THEN 0.2 ELSE 0.0 END +
                    CASE WHEN h.score > 0 THEN 0.1 ELSE 0.0 END
                ) >= 0.7 THEN 'high'
                WHEN (
                    COALESCE(1.0 - cf.coverage_ratio, 1.0) * 0.4 +
                    CASE fm.complexity_bucket
                        WHEN 'high' THEN 0.4
                        WHEN 'medium' THEN 0.2
                        ELSE 0.0
                    END +
                    CASE WHEN sd.has_errors THEN 0.2 ELSE 0.0 END +
                    CASE WHEN h.score > 0 THEN 0.1 ELSE 0.0 END
                ) >= 0.4 THEN 'medium'
                ELSE 'low'
            END AS risk_level,
            m.tags,
            m.owners,
            NOW() AS created_at
        FROM analytics.function_metrics fm
        LEFT JOIN analytics.function_types ft
            ON ft.function_goid_h128 = fm.function_goid_h128
        LEFT JOIN analytics.coverage_functions cf
            ON cf.function_goid_h128 = fm.function_goid_h128
        LEFT JOIN analytics.hotspots h
            ON h.rel_path = fm.rel_path
        LEFT JOIN analytics.typedness ty
            ON ty.path = fm.rel_path
        LEFT JOIN analytics.static_diagnostics sd
            ON sd.rel_path = fm.rel_path
        LEFT JOIN (
            SELECT
                function_goid_h128,
                COUNT(*) AS test_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failing_test_count,
                any_value(status IGNORE NULLS) AS last_test_status
            FROM coverage.test_edges
            WHERE repo = ? AND commit = ?
            GROUP BY function_goid_h128
        ) t_stats
            ON t_stats.function_goid_h128 = fm.function_goid_h128
        LEFT JOIN (
            SELECT module AS path, tags, owners FROM temp.catalog_modules
        ) m
            ON m.path = fm.rel_path
    """

    con.execute(
        "DELETE FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    )
    con.execute(risk_sql, [ctx.repo, ctx.commit])

    if use_catalog_modules:
        refresh_subsystem_caches(
            ctx.gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            benchmark=False,
        )
    return None


RISK_FACTORS_PLUGIN = AnalyticsPlugin(
    name="risk_factors.build",
    description="Aggregate analytics into per-function risk scores and levels.",
    stage="profiles",
    enabled_by_default=True,
    run=_risk_factors_run,
    severity="fatal",
    depends_on=(
        "functions.metrics",
        "coverage.functions",
        "hotspots.build",
        "typing_ingest",
        "tests_ingest",
        "coverage.test_edges",
        "config_ingest",
    ),
    provides=("analytics.goid_risk_factors",),
    requires=("analytics.function_metrics",),
    resource_hints=ResourceHints(max_runtime_ms=90_000, priority=60),
    row_count_tables=("analytics.goid_risk_factors",),
)

register_analytics_plugin(RISK_FACTORS_PLUGIN)


__all__ = ["RISK_FACTORS_PLUGIN"]
