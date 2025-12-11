"""Symbol graph metrics plugin.

Compute graph metrics from symbol usage patterns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.graphs.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


SYMBOL_GRAPH_METRICS_METADATA = CorePluginMetadata(
    name="analytics.symbol_graph_metrics",
    version="3.0.0",
    description="Compute graph metrics from symbol usage patterns.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="symbol",
    provides=(
        "analytics.symbol_graph_metrics_functions",
        "analytics.symbol_graph_metrics_modules",
    ),
    requires=("graph.symbol_use_edges",),
    produces_tables=(
        "analytics.symbol_graph_metrics_functions",
        "analytics.symbol_graph_metrics_modules",
    ),
    consumes_tables=("graph.symbol_use_edges",),
)


class SymbolGraphMetricsPlugin(TargetPlugin):
    """Compute graph metrics from symbol usage patterns.

    Analyzes symbol definition-to-use relationships to compute:
    - Symbol coupling metrics
    - Cross-module symbol flow
    - Symbol centrality measures

    Outputs
    -------
    - analytics.symbol_graph_metrics_functions: Per-function symbol metrics
    - analytics.symbol_graph_metrics_modules: Per-module symbol metrics
    """

    plugin_name: ClassVar[str] = "symbol_graph_metrics"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Compute graph metrics from symbol usage patterns."
    _core_metadata: ClassVar[CorePluginMetadata] = SYMBOL_GRAPH_METRICS_METADATA

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

        row_counts: dict[str, int] = {}

        # Compute module-level metrics
        try:
            log.info("Computing symbol graph metrics (modules) for %s@%s", repo, commit)
            compute_symbol_graph_metrics_modules(
                ctx.gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
            row = ctx.gateway.con.execute(
                """
                SELECT COUNT(*) FROM analytics.symbol_graph_metrics_modules
                WHERE repo = ? AND commit = ?
                """,
                [repo, commit],
            ).fetchone()
            row_counts["analytics.symbol_graph_metrics_modules"] = int(row[0]) if row else 0
        except (RuntimeError, ValueError, OSError) as e:
            log.warning("Symbol graph metrics (modules) failed: %s", e)

        # Compute function-level metrics
        try:
            log.info("Computing symbol graph metrics (functions) for %s@%s", repo, commit)
            compute_symbol_graph_metrics_functions(
                ctx.gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
            row = ctx.gateway.con.execute(
                """
                SELECT COUNT(*) FROM analytics.symbol_graph_metrics_functions
                WHERE repo = ? AND commit = ?
                """,
                [repo, commit],
            ).fetchone()
            row_counts["analytics.symbol_graph_metrics_functions"] = int(row[0]) if row else 0
        except (RuntimeError, ValueError, OSError) as e:
            log.warning("Symbol graph metrics (functions) failed: %s", e)

        log.info("Symbol graph metrics completed: %s", row_counts)
        return TargetResult.succeeded(row_counts=row_counts)


__all__ = ["SYMBOL_GRAPH_METRICS_METADATA", "SymbolGraphMetricsPlugin"]
