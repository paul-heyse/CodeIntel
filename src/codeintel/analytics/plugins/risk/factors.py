"""Risk factors plugin using the new protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from codeintel.analytics.resources.catalog import CatalogProvider

from codeintel.analytics.core.context import PluginExecutionContext
from codeintel.analytics.core.protocol import (
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.subsystems import refresh_subsystem_caches
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


@dataclass
class RiskFactorsPlugin:
    """Plugin for aggregating risk factors.

    Computes per-function risk scores based on:
    - Code complexity
    - Coverage metrics
    - Hotspot patterns
    - Static analysis findings
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="risk_factors.build",
            description="Aggregate analytics into per-function risk scores and levels.",
            kind="analytics",
            stage="risk",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(),
            outputs=(
                PluginOutputSpec(name="goid_risk_factors", tables=("analytics.goid_risk_factors",)),
            ),
            provides=("analytics.goid_risk_factors",),
            requires=("analytics.function_metrics",),
            depends_on=(
                "functions.metrics",
                "coverage.functions",
                "hotspots.build",
                "typing_ingest",
                "tests_ingest",
                "coverage.test_edges",
                "config_ingest",
            ),
            resource_hints=PluginResourceHints(
                max_runtime_ms=90_000,
                priority=60,
            ),
            tags=("risk", "factors", "scoring"),
        )

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate required inputs.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        ValidationResult
            Always returns success as this plugin has no required configs.
        """
        _ = self.metadata
        _ = ctx  # No validation needed for this plugin
        return ValidationResult.success()

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        PluginResult
            Execution result.
        """
        _ = self.metadata
        gateway = ctx.gateway
        con = gateway.con

        catalog = None
        if ctx.has_resource_by_name("CatalogProvider"):
            catalog_resource = cast("CatalogProvider", ctx.require_by_name("CatalogProvider"))
            catalog = catalog_resource.get()

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
            return PluginResult.fail(f"Risk factors computation failed: {e}")

        return PluginResult.ok()


__all__ = ["RiskFactorsPlugin"]
