"""Ensure compute_graph_metrics reuses provided runtime/engine."""

from __future__ import annotations

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.analytics.graphs.graph_metrics import GraphMetricsDeps, compute_graph_metrics
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from tests._helpers.fixtures import ProvisionedGateway


def test_compute_graph_metrics_reuses_runtime_engine(provisioned_repo: ProvisionedGateway) -> None:
    """
    Provided GraphRuntime should be reused without rebuilding the engine.

    Raises
    ------
    AssertionError
        When the runtime engine is unexpectedly absent.
    """
    gateway = provisioned_repo.gateway
    snapshot = SnapshotRef(
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        repo_root=provisioned_repo.repo_root,
    )
    runtime = build_graph_runtime(
        gateway,
        GraphRuntimeOptions(snapshot=snapshot),
    )
    compute_graph_metrics(
        gateway,
        GraphMetricsStepConfig(snapshot=snapshot),
        deps=GraphMetricsDeps(runtime=runtime),
    )
    if runtime.engine is None:
        message = "Runtime engine should be populated"
        raise AssertionError(message)
