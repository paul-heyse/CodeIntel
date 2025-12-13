"""Risk factors plugin.

This plugin aggregates analytics into per-function risk scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.subsystems import refresh_subsystem_caches
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway


def _seed_catalog_modules(
    gateway: StorageGateway,
    catalog: FunctionCatalogProvider | None,
    *,
    repo: str,
    commit: str,
) -> bool:
    """Create a temporary table of modules from a catalog when available.

    Parameters
    ----------
    gateway
        Storage gateway.
    catalog
        Function catalog provider.
    repo
        Repository identifier.
    commit
        Commit identifier.

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


RISK_FACTORS_METADATA = CorePluginMetadata(
    name="analytics.goid_risk_factors",
    version="3.0.0",
    description="Aggregate analytics into per-function risk scores and levels.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="risk",
    provides=("analytics.goid_risk_factors",),
    requires=(
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.coverage_functions",
        "analytics.hotspots",
        "analytics.typedness",
        "analytics.static_diagnostics",
        "analytics.test_coverage_edges",
    ),
    produces_tables=("analytics.goid_risk_factors",),
    consumes_tables=(
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.coverage_functions",
        "analytics.hotspots",
        "analytics.typedness",
        "analytics.static_diagnostics",
        "analytics.test_coverage_edges",
        "core.modules",
    ),
)


class RiskFactorsPlugin(MetadataPlugin):
    """Aggregate analytics into per-function risk scores and levels.

    Computes per-function risk scores based on:
    - Code complexity
    - Coverage metrics
    - Hotspot patterns
    - Static analysis findings

    Outputs
    -------
    - analytics.goid_risk_factors: Per-function risk scores
    """

    _core_metadata: ClassVar[CorePluginMetadata] = RISK_FACTORS_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result.
        """
        _ = self

        gateway = ctx.gateway
        con = gateway.con
        catalog = ctx.resources.catalog

        try:
            con.execute(
                "DELETE FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?",
                [ctx.repo, ctx.commit],
            )

            use_catalog_modules = _seed_catalog_modules(
                gateway, catalog, repo=ctx.repo, commit=ctx.commit
            )

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
                        SUM(CASE WHEN last_status = 'failed' THEN 1 ELSE 0 END)
                            AS failing_test_count,
                        any_value(last_status) AS last_test_status
                    FROM analytics.test_coverage_edges
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
            if not use_catalog_modules:
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
            con.execute(risk_sql, [ctx.repo, ctx.commit])

            if use_catalog_modules:
                refresh_subsystem_caches(
                    ctx.gateway,
                    repo=ctx.repo,
                    commit=ctx.commit,
                    benchmark=False,
                )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Risk factors computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["RISK_FACTORS_METADATA", "RiskFactorsPlugin"]
