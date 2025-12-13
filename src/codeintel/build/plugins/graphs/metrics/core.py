"""Core graph metrics plugin.

This module computes core graph metrics (PageRank, centrality, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.graphs import (
    compute_graph_metrics,
    compute_graph_metrics_functions_ext,
    compute_graph_metrics_modules_ext,
    compute_graph_stats,
)
from codeintel.analytics.graphs.graph_metrics import GraphMetricsDeps
from codeintel.analytics.runtime import (
    GraphMetricsOptions,
    GraphRuntimeOptions,
    build_graph_runtime,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config.primitives import GraphBackendConfig
from codeintel.storage.ibis_types import and_predicates

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


class CoreMetricsPlugin(TargetPlugin):
    """Compute core graph metrics (PageRank, centrality, etc.).

    Outputs
    -------
    - analytics.graph_metrics_functions: Function-level graph metrics
    - analytics.graph_metrics_modules: Module-level graph metrics
    - analytics.graph_metrics_functions_ext: Extended function metrics
    - analytics.graph_metrics_modules_ext: Extended module metrics
    """

    plugin_name: ClassVar[str] = "graph_metrics"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Compute core graph metrics (PageRank, centrality, etc.)."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute core metrics computation.

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
        snapshot = ctx.snapshot
        repo, commit = snapshot.repo, snapshot.commit

        try:
            log.info(
                "core_metrics.execute repo=%s commit=%s",
                repo,
                commit,
            )

            backend_config = GraphBackendConfig(use_gpu=True, backend="auto", strict=False)
            runtime_options = GraphRuntimeOptions(snapshot=snapshot, backend=backend_config)
            runtime = build_graph_runtime(ctx.gateway, runtime_options)

            options = GraphMetricsOptions()
            deps = GraphMetricsDeps(
                catalog_provider=ctx.resources.catalog,
                runtime=runtime,
            )
            compute_graph_metrics(ctx.gateway, snapshot, options=options, deps=deps)

            compute_graph_metrics_functions_ext(
                ctx.gateway,
                repo=repo,
                commit=commit,
                runtime=runtime,
            )

            compute_graph_metrics_modules_ext(
                ctx.gateway,
                repo=repo,
                commit=commit,
                runtime=runtime,
            )

            compute_graph_stats(
                ctx.gateway,
                repo=repo,
                commit=commit,
                runtime=runtime,
            )

            row_counts: dict[str, int] = {}
            for table in [
                "analytics.graph_metrics_functions",
                "analytics.graph_metrics_modules",
                "analytics.graph_metrics_functions_ext",
                "analytics.graph_metrics_modules_ext",
                "analytics.graph_stats",
            ]:
                expr = ctx.gateway.ibis.table(table)
                filtered = expr.filter(
                    and_predicates(expr.repo == repo, expr.commit == commit)
                )
                result_df = filtered.aggregate(row_count=expr.repo.count()).execute()
                row_counts[table] = (
                    int(result_df.iloc[0]["row_count"]) if not result_df.empty else 0
                )

            log.info("core_metrics.complete row_counts=%s", row_counts)
            return TargetResult.succeeded(row_counts=row_counts)
        except (RuntimeError, ValueError, OSError) as e:
            log.exception("core_metrics.failed")
            return TargetResult.failed(f"Core metrics computation failed: {e}")


__all__ = ["CoreMetricsPlugin"]
