"""Test graph metrics plugin.

Compute graph metrics from the test-function bipartite graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.testing.graph_metrics import compute_test_graph_metrics
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


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
            log.info("Computing test graph metrics for %s@%s", repo, commit)
            compute_test_graph_metrics(
                ctx.gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Test graph metrics computation failed: {e}")

        # Count rows written
        row_counts: dict[str, int] = {}
        for table in (
            "analytics.test_graph_metrics_tests",
            "analytics.test_graph_metrics_functions",
        ):
            row = ctx.gateway.con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE repo = ? AND commit = ?",  # noqa: S608
                [repo, commit],
            ).fetchone()
            row_counts[table] = int(row[0]) if row else 0

        log.info("Test graph metrics completed: %s", row_counts)
        return TargetResult.succeeded(row_counts=row_counts)


__all__ = ["TestGraphMetricsPlugin"]
