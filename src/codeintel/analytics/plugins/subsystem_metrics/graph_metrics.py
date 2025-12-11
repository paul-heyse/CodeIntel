"""Subsystem graph metrics plugin.

Compute graph metrics for subsystems.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.graphs.subsystem_graph_metrics import (
    compute_subsystem_graph_metrics,
)
from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata

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


class SubsystemGraphMetricsPlugin(TargetPlugin):
    """Compute graph metrics for subsystems.

    Analyzes the condensed import graph at the subsystem level:
    - Subsystem coupling metrics
    - Inter-subsystem dependencies
    - Subsystem centrality measures

    Outputs
    -------
    - analytics.subsystem_graph_metrics: Per-subsystem graph metrics
    """

    plugin_name: ClassVar[str] = "subsystem_graph_metrics"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Compute graph metrics for subsystems."
    _core_metadata: ClassVar[CorePluginMetadata] = SUBSYSTEM_GRAPH_METRICS_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

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
        _ = self  # Protocol method requires instance

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

        # Count rows written
        row = ctx.gateway.con.execute(
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
