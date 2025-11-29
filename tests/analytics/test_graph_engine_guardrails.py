"""Guardrails to ensure graph engines are not rebuilt once a runtime is provided."""

from __future__ import annotations

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.analytics.graph_service_runtime import GraphServiceRuntime
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from tests._helpers.fixtures import ProvisionedGateway


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


def test_graph_metrics_reuse_provided_runtime(provisioned_repo: ProvisionedGateway) -> None:
    """Graph metric orchestration should reuse the provided runtime engine."""
    gateway = provisioned_repo.gateway
    snapshot = SnapshotRef(
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        repo_root=provisioned_repo.repo_root,
    )
    runtime = build_graph_runtime(gateway, GraphRuntimeOptions(snapshot=snapshot))

    service = GraphServiceRuntime(gateway=gateway, runtime=runtime)
    service.run_plugins(("core_graph_metrics",), cfg=GraphMetricsStepConfig(snapshot=snapshot))

    _expect(
        condition=runtime.engine is not None,
        detail="Runtime engine must remain populated after metrics run",
    )
