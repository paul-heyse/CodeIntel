"""Test graph metrics plugin.

Compute graph metrics from the test-function bipartite graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.testing.graph_metrics import compute_test_graph_metrics
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugins._metadata import to_plugin_metadata
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.storage.ibis_types import and_predicates

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata

log = logging.getLogger(__name__)


TEST_GRAPH_METRICS_METADATA = CorePluginMetadata(
    name="analytics.test_graph_metrics",
    version="3.0.0",
    description="Compute graph metrics from the test-function bipartite graph.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="test",
    provides=(
        "analytics.test_graph_metrics_tests",
        "analytics.test_graph_metrics_functions",
    ),
    requires=("analytics.test_coverage_edges",),
    produces_tables=(
        "analytics.test_graph_metrics_tests",
        "analytics.test_graph_metrics_functions",
    ),
    consumes_tables=("analytics.test_coverage_edges",),
)


class TestGraphMetricsPlugin(TargetPlugin):
    """Compute graph metrics from the test-function bipartite graph.

    Analyzes the relationship between tests and the functions they cover
    to compute metrics like:
    - PageRank and betweenness centrality
    - Test clustering and coverage spread
    - Function coverage depth

    Outputs
    -------
    - analytics.test_graph_metrics_tests: Per-test graph metrics
    - analytics.test_graph_metrics_functions: Per-function test coverage metrics
    """

    plugin_name: ClassVar[str] = "test_graph_metrics"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Compute graph metrics from the test-function bipartite graph."
    )
    _core_metadata: ClassVar[CorePluginMetadata] = TEST_GRAPH_METRICS_METADATA

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
        _ = self

        repo = ctx.snapshot.repo
        commit = ctx.snapshot.commit
        graph_runtime = ctx.resources.graph_runtime

        try:
            log.info("Computing test graph metrics for %s@%s", repo, commit)
            compute_test_graph_metrics(
                ctx.gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Test graph metrics computation failed: {e}")

        row_counts: dict[str, int] = {}
        for table in (
            "analytics.test_graph_metrics_tests",
            "analytics.test_graph_metrics_functions",
        ):
            expr = ctx.gateway.ibis.table(table)
            filtered = expr.filter(and_predicates(expr.repo == repo, expr.commit == commit))
            result_df = filtered.aggregate(row_count=expr.repo.count()).execute()
            row_counts[table] = int(result_df.iloc[0]["row_count"]) if not result_df.empty else 0

        log.info("Test graph metrics completed: %s", row_counts)
        return TargetResult.succeeded(row_counts=row_counts)


__all__ = ["TEST_GRAPH_METRICS_METADATA", "TestGraphMetricsPlugin"]
