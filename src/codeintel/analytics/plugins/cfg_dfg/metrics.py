"""CFG/DFG metrics plugin.

Compute control-flow and data-flow graph metrics per function.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.plugins._metadata import to_plugin_metadata
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.storage.ibis_types import and_predicates

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata

log = logging.getLogger(__name__)


CFG_DFG_METRICS_METADATA = CorePluginMetadata(
    name="analytics.cfg_dfg_metrics",
    version="3.0.0",
    description="Compute control-flow and data-flow graph metrics per function.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="cfg",
    provides=(
        "analytics.cfg_function_metrics",
        "analytics.cfg_block_metrics",
        "analytics.cfg_function_metrics_ext",
        "analytics.dfg_function_metrics",
        "analytics.dfg_block_metrics",
        "analytics.dfg_function_metrics_ext",
    ),
    requires=("graph.cfg_blocks", "graph.cfg_edges", "graph.dfg_edges"),
    produces_tables=(
        "analytics.cfg_function_metrics",
        "analytics.cfg_block_metrics",
        "analytics.cfg_function_metrics_ext",
        "analytics.dfg_function_metrics",
        "analytics.dfg_block_metrics",
        "analytics.dfg_function_metrics_ext",
    ),
    consumes_tables=("graph.cfg_blocks", "graph.cfg_edges", "graph.dfg_edges"),
)


class CfgDfgMetricsPlugin(TargetPlugin):
    """Compute CFG and DFG metrics per function.

    Analyzes control-flow graphs and data-flow graphs to produce
    function-level and block-level metrics including:
    - Block counts, edge counts, cycle detection
    - Centrality measures (betweenness, closeness, eigenvector)
    - Loop analysis (nesting depth, headers)
    - Dominance analysis (dominator tree, frontiers)

    Outputs
    -------
    - analytics.cfg_function_metrics: CFG metrics per function
    - analytics.cfg_block_metrics: CFG metrics per block
    - analytics.cfg_function_metrics_ext: Extended CFG metrics
    - analytics.dfg_function_metrics: DFG metrics per function
    - analytics.dfg_block_metrics: DFG metrics per block
    - analytics.dfg_function_metrics_ext: Extended DFG metrics
    """

    plugin_name: ClassVar[str] = "cfg_dfg_metrics"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Compute control-flow and data-flow graph metrics per function."
    )
    _core_metadata: ClassVar[CorePluginMetadata] = CFG_DFG_METRICS_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute CFG/DFG metrics computation.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result with row counts.
        """
        _ = self

        repo = ctx.snapshot.repo
        commit = ctx.snapshot.commit

        row_counts: dict[str, int] = {}

        def _count_rows(table_key: str) -> int:
            table = ctx.gateway.ibis.table(table_key)
            filtered = table.filter(and_predicates(table.repo == repo, table.commit == commit))
            result_df = filtered.aggregate(row_count=table.repo.count()).execute()
            return int(result_df.iloc[0]["row_count"]) if not result_df.empty else 0

        try:
            log.info("Computing CFG metrics for %s@%s", repo, commit)
            compute_cfg_metrics(ctx.gateway, repo=repo, commit=commit)

            for table in (
                "analytics.cfg_function_metrics",
                "analytics.cfg_block_metrics",
                "analytics.cfg_function_metrics_ext",
            ):
                row_counts[table] = _count_rows(table)
        except (RuntimeError, ValueError, OSError) as e:
            log.warning("CFG metrics computation failed: %s", e)

        try:
            log.info("Computing DFG metrics for %s@%s", repo, commit)
            compute_dfg_metrics(ctx.gateway, repo=repo, commit=commit)

            for table in (
                "analytics.dfg_function_metrics",
                "analytics.dfg_block_metrics",
                "analytics.dfg_function_metrics_ext",
            ):
                row_counts[table] = _count_rows(table)
        except (RuntimeError, ValueError, OSError) as e:
            log.warning("DFG metrics computation failed: %s", e)

        total_rows = sum(row_counts.values())
        log.info("CFG/DFG metrics completed: %d total rows", total_rows)

        return TargetResult.succeeded(row_counts=row_counts)


__all__ = ["CFG_DFG_METRICS_METADATA", "CfgDfgMetricsPlugin"]
