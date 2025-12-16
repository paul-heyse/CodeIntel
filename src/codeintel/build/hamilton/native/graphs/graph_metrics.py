"""Native Hamilton implementation for graph_metrics target.

This module implements graph metrics computation as a native Hamilton pipeline with:
- t__graph_metrics__compute: Compute graph metrics using analytics module
- t__graph_metrics: Materialize with validators and return TargetRunRecord

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.analytics.graphs import (
    compute_graph_metrics,
    compute_graph_metrics_functions_ext,
    compute_graph_metrics_modules_ext,
    compute_graph_stats,
)
from codeintel.analytics.graphs.graph_metrics import GraphMetricsDeps
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.config.primitives import GraphBackendConfig
from codeintel.graphs.runtime import (
    GraphMetricsOptions,
    GraphRuntimeOptions,
    build_graph_runtime,
)
from codeintel.storage.ibis_types import and_predicates

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

_OUTPUT_TABLES = (
    "analytics.graph_metrics_functions",
    "analytics.graph_metrics_modules",
    "analytics.graph_metrics_functions_ext",
    "analytics.graph_metrics_modules_ext",
    "analytics.graph_stats",
)


@dataclass(frozen=True)
class GraphMetricsComputeResult:
    """Result from graph metrics computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    table_counts
        Row counts per produced table.
    error
        Fatal error message if computation failed.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@tag(node_type="helper")
def _count_rows(
    gateway: StorageGateway,
    table: str,
    repo: str,
    commit: str,
) -> int:
    """Count rows in a table for the given snapshot.

    Parameters
    ----------
    gateway
        Storage gateway.
    table
        Table name.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Row count.
    """
    try:
        tbl = gateway.ibis.table(table)
        filtered = tbl.filter(and_predicates(tbl.repo == repo, tbl.commit == commit))
        result_df = filtered.aggregate(row_count=tbl.repo.count()).execute()
        return int(result_df.iloc[0]["row_count"]) if not result_df.empty else 0
    except (RuntimeError, ValueError, OSError, KeyError):
        return 0


@tag(domain="graphs", target="graph_metrics", node_type="compute")
def t__graph_metrics__compute(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
) -> GraphMetricsComputeResult:
    """Compute graph metrics from call graph data.

    This is the compute node for the graph_metrics target. It computes
    PageRank, centrality, and other graph metrics using NetworkX.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__call_graph
        Upstream call_graph target result (for dependency).

    Returns
    -------
    GraphMetricsComputeResult
        Result containing table row counts.

    Notes
    -----
    Produces:
    - analytics.graph_metrics_functions: Function-level graph metrics
    - analytics.graph_metrics_modules: Module-level graph metrics
    - analytics.graph_metrics_functions_ext: Extended function metrics
    - analytics.graph_metrics_modules_ext: Extended module metrics
    - analytics.graph_stats: Overall graph statistics
    """
    if t__call_graph.status != "succeeded":
        return GraphMetricsComputeResult(
            success=False,
            error=f"Upstream call_graph target failed: {t__call_graph.error}",
        )

    try:
        gateway = env.gateway
        snapshot = env.snapshot
        repo, commit = snapshot.repo, snapshot.commit

        log.info(
            "graph_metrics: Computing metrics for repo=%s commit=%s",
            repo,
            commit,
        )

        backend_config = GraphBackendConfig(use_gpu=True, backend="auto", strict=False)
        runtime_options = GraphRuntimeOptions(snapshot=snapshot, backend=backend_config)
        runtime = build_graph_runtime(gateway, runtime_options)

        options = GraphMetricsOptions()
        deps = GraphMetricsDeps(
            catalog_provider=None,
            runtime=runtime,
        )
        compute_graph_metrics(gateway, snapshot, options=options, deps=deps)

        compute_graph_metrics_functions_ext(
            gateway,
            repo=repo,
            commit=commit,
            runtime=runtime,
        )

        compute_graph_metrics_modules_ext(
            gateway,
            repo=repo,
            commit=commit,
            runtime=runtime,
        )

        compute_graph_stats(
            gateway,
            repo=repo,
            commit=commit,
            runtime=runtime,
        )

        row_counts: dict[str, int] = {}
        for table in _OUTPUT_TABLES:
            row_counts[table] = _count_rows(gateway, table, repo, commit)

        log.info("graph_metrics: Computed metrics row_counts=%s", row_counts)

        return GraphMetricsComputeResult(
            success=True,
            table_counts=row_counts,
        )

    except (RuntimeError, ValueError, OSError) as exc:
        log.exception("Graph metrics computation failed")
        return GraphMetricsComputeResult(
            success=False,
            error=str(exc),
        )


@tag(domain="graphs", target="graph_metrics", node_type="materialize")
def t__graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__graph_metrics__compute: GraphMetricsComputeResult,
) -> TargetRunRecord:
    """Materialize graph metrics target with validation.

    This is the entry point for the graph_metrics target. It orchestrates
    graph metrics computation and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__graph_metrics__compute
        Computation result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "graph_metrics")

    if executor.should_skip():
        return executor.skip()

    if not t__graph_metrics__compute.success:
        return executor.fail(
            RuntimeError(t__graph_metrics__compute.error or "Graph metrics computation failed")
        )

    def compute() -> dict[str, int]:
        return dict(t__graph_metrics__compute.table_counts)

    return executor.execute(compute)


__all__ = [
    "GraphMetricsComputeResult",
    "t__graph_metrics",
    "t__graph_metrics__compute",
]
