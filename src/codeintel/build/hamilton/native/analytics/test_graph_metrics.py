"""Native Hamilton implementation for test_graph_metrics target.

This module provides the Hamilton native nodes for test graph metrics:
- `t__test_graph_metrics__compute`: Pure compute node for test graph metrics
- `t__test_graph_metrics`: Materialize node that writes both tables

The compute node calls pure functions from `codeintel.analytics.testing.compute`
which return structured result containers. The materialize node uses
`materialize_rows` to persist the data to DuckDB with proper asset tracking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.analytics.testing.compute import compute_test_graph_metrics_pure
from codeintel.analytics.testing.graph_metrics import (
    TEST_GRAPH_METRICS_FUNCTIONS_COLS,
    TEST_GRAPH_METRICS_TESTS_COLS,
)
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.graphs.runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

if TYPE_CHECKING:
    from codeintel.analytics.testing.compute import TestGraphMetricsResult
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.manifest_hook import TargetRunRecord
    from codeintel.build.targets import TargetGraph
    from codeintel.graphs.runtime import GraphRuntime


log = logging.getLogger(__name__)


def _get_graph_runtime(env: BuildEnv) -> GraphRuntime:
    """Get graph runtime from build environment.

    Resolves the graph runtime from the build environment, creating one
    if not already available.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    GraphRuntime
        Resolved graph runtime for the snapshot.
    """
    return resolve_graph_runtime(env.gateway, env.snapshot, GraphRuntimeOptions())


@tag(domain="analytics", target="test_graph_metrics", node_type="compute")
def t__test_graph_metrics__compute(env: BuildEnv) -> TestGraphMetricsResult:
    """Compute test graph metrics for all tests and functions.

    This is a pure compute node with no side effects. It computes graph
    metrics from the test-function bipartite graph and returns row data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    TestGraphMetricsResult
        Container with rows for test and function metrics tables.

    Notes
    -----
    The metrics computed include:
    - Degree and weighted degree in the bipartite graph
    - Degree centrality
    - Projection metrics (clustering, betweenness)
    - Risk-weighted degree based on function risk scores
    """
    runtime = _get_graph_runtime(env)
    return compute_test_graph_metrics_pure(env.gateway, env.snapshot, runtime)


@tag(domain="analytics", target="test_graph_metrics", node_type="materialize")
def t__test_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__test_graph_metrics__compute: TestGraphMetricsResult,
) -> TargetRunRecord:
    """Materialize both test graph metrics tables to DuckDB.

    This is the only side-effect boundary for this target. It writes
    the computed metrics to DuckDB and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__test_graph_metrics__compute
        Computed test graph metrics from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables:
    - analytics.test_graph_metrics_tests
    - analytics.test_graph_metrics_functions
    """
    executor = NativeTargetExecutor.for_target(env, graph, "test_graph_metrics")

    if executor.should_skip():
        return executor.skip()

    def compute() -> dict[str, int]:
        # Ensure tables exist
        backend = DuckDBPolicyBackend(env.gateway)
        backend.ensure_table("analytics.test_graph_metrics_tests")
        backend.ensure_table("analytics.test_graph_metrics_functions")

        ctx = MaterializationContext(
            gateway=env.gateway,
            snapshot=env.snapshot,
            validate=env.validate_outputs,
            owner_target="test_graph_metrics",
            input_hash=executor.input_hash,
        )

        row_counts: dict[str, int] = {}

        # Materialize test metrics table
        test_ref = materialize_rows(
            ctx,
            "analytics.test_graph_metrics_tests",
            t__test_graph_metrics__compute.test_rows,
            TEST_GRAPH_METRICS_TESTS_COLS,
        )
        row_counts["analytics.test_graph_metrics_tests"] = test_ref.row_count or 0

        # Materialize function metrics table
        func_ref = materialize_rows(
            ctx,
            "analytics.test_graph_metrics_functions",
            t__test_graph_metrics__compute.function_rows,
            TEST_GRAPH_METRICS_FUNCTIONS_COLS,
        )
        row_counts["analytics.test_graph_metrics_functions"] = func_ref.row_count or 0

        return row_counts

    return executor.execute(compute)


# Export node names for Hamilton discovery
__all__ = [
    "t__test_graph_metrics",
    "t__test_graph_metrics__compute",
]
