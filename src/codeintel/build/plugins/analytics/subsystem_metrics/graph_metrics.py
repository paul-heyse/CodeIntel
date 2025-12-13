"""Subsystem graph metrics plugin.

Compute graph metrics for subsystems.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.graphs.subsystem_graph_metrics import (
    compute_subsystem_graph_metrics,
)
from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


SUBSYSTEM_GRAPH_METRICS_METADATA = CorePluginMetadata(
    name="analytics.subsystem_graph_metrics",
    version="3.0.0",
    description="Compute graph metrics for subsystems.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="subsystem",
    provides=("analytics.subsystem_graph_metrics",),
    requires=("analytics.subsystems", "analytics.subsystem_modules"),
    produces_tables=("analytics.subsystem_graph_metrics",),
    consumes_tables=("analytics.subsystems", "analytics.subsystem_modules"),
)


class SubsystemGraphMetricsPlugin(MetadataPlugin):
    """Compute graph metrics for subsystems.

    Analyzes the condensed import graph at the subsystem level:
    - Subsystem coupling metrics
    - Inter-subsystem dependencies
    - Subsystem centrality measures

    Outputs
    -------
    - analytics.subsystem_graph_metrics: Per-subsystem graph metrics
    """

    _core_metadata: ClassVar[CorePluginMetadata] = SUBSYSTEM_GRAPH_METRICS_METADATA

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

        repo = ctx.snapshot.repo
        commit = ctx.snapshot.commit
        graph_runtime = ctx.resources.graph_runtime

        try:
            log.info("Computing subsystem graph metrics for %s@%s", repo, commit)
            compute_subsystem_graph_metrics(
                ctx.gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Subsystem graph metrics failed: {e}")

        row = ctx.gateway.execute(
            """
            SELECT COUNT(*) FROM analytics.subsystem_graph_metrics
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchone()
        row_count = int(row[0]) if row else 0

        log.info("Subsystem graph metrics completed: %d rows", row_count)
        return TargetResult.succeeded(row_counts={"analytics.subsystem_graph_metrics": row_count})


__all__ = ["SUBSYSTEM_GRAPH_METRICS_METADATA", "SubsystemGraphMetricsPlugin"]
