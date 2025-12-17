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
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.testing.compute import (
    TestGraphMetricsResult,
    compute_test_graph_metrics_pure,
)
from codeintel.analytics.testing.graph_metrics import (
    TEST_GRAPH_METRICS_FUNCTIONS_COLS,
    TEST_GRAPH_METRICS_TESTS_COLS,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.graphs.runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    GraphRuntime,
    TargetGraph,
    TargetRunRecord,
    TestGraphMetricsResult,
)


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
def t__test_graph_metrics__compute(env: BuildEnv, graph: TargetGraph) -> TestGraphMetricsResult | None:
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
    target = graph.get("test_graph_metrics")
    if target is not None:
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=None,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

    runtime = _get_graph_runtime(env)
    return compute_test_graph_metrics_pure(env.gateway, env.snapshot, runtime)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.test_graph_metrics_tests"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("test_graph_metrics"),
    table_key=value("analytics.test_graph_metrics_tests"),
    columns=value(tuple(TEST_GRAPH_METRICS_TESTS_COLS)),
)
@tag(domain="analytics", target="test_graph_metrics", node_type="compute", target_="test_graph_metrics__tests_rows")
def test_graph_metrics__tests_rows(
    t__test_graph_metrics__compute: TestGraphMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_graph_metrics_tests."""
    if t__test_graph_metrics__compute is None:
        return None
    return tuple(t__test_graph_metrics__compute.test_rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.test_graph_metrics_functions"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("test_graph_metrics"),
    table_key=value("analytics.test_graph_metrics_functions"),
    columns=value(tuple(TEST_GRAPH_METRICS_FUNCTIONS_COLS)),
)
@tag(
    domain="analytics",
    target="test_graph_metrics",
    node_type="compute",
    target_="test_graph_metrics__functions_rows",
)
def test_graph_metrics__functions_rows(
    t__test_graph_metrics__compute: TestGraphMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_graph_metrics_functions."""
    if t__test_graph_metrics__compute is None:
        return None
    return tuple(t__test_graph_metrics__compute.function_rows)


@tag(domain="analytics", target="test_graph_metrics", node_type="materialize")
def t__test_graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__test_graph_metrics_tests: dict[str, Any],
    m__analytics__test_graph_metrics_functions: dict[str, Any],
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
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name="test_graph_metrics",
        materializations={
            "analytics.test_graph_metrics_tests": m__analytics__test_graph_metrics_tests,
            "analytics.test_graph_metrics_functions": m__analytics__test_graph_metrics_functions,
        },
    )


# Export node names for Hamilton discovery
__all__ = [
    "t__test_graph_metrics",
    "t__test_graph_metrics__compute",
]
