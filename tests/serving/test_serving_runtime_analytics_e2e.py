"""E2E guardrails spanning serving wiring through analytics metrics."""

from __future__ import annotations

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    GraphRuntimePool,
    build_graph_runtime,
)
from codeintel.analytics.graph_service_runtime import GraphServiceRuntime
from codeintel.config.primitives import SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from codeintel.serving.services.wiring import BackendResourceOptions, build_backend_resource
from codeintel.storage.gateway import StorageGateway
from tests._helpers.fixtures import ProvisionedGateway


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


def _make_snapshot(ctx: ProvisionedGateway) -> SnapshotRef:
    return SnapshotRef(repo=ctx.repo, commit=ctx.commit, repo_root=ctx.repo_root)


class TrackingGraphRuntimePool(GraphRuntimePool):
    """GraphRuntimePool that records get() invocations for assertions."""

    def __init__(self, max_size: int = 4) -> None:
        super().__init__(max_size=max_size)
        self.get_calls = 0

    def get(
        self,
        gateway: StorageGateway,
        options: GraphRuntimeOptions,
        *,
        context: AnalyticsContext | None = None,
    ) -> GraphRuntime:
        """
        Return a runtime while incrementing call counters for assertions.

        Returns
        -------
        GraphRuntime
            Runtime acquired from the pool.
        """
        self.get_calls += 1
        return super().get(gateway, options, context=context)


def test_serving_runtime_analytics_reuses_engine(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Serving wiring plus analytics metrics should reuse a single runtime engine."""
    gateway = provisioned_repo.gateway
    snapshot = _make_snapshot(provisioned_repo)
    cfg = ServingConfig(
        mode="local_db",
        repo_root=snapshot.repo_root,
        repo=snapshot.repo,
        commit=snapshot.commit,
        db_path=gateway.config.db_path,
    )
    pool = TrackingGraphRuntimePool(max_size=2)
    resource = build_backend_resource(
        cfg,
        gateway=gateway,
        options=BackendResourceOptions(runtime_pool=pool),
    )

    runtime = pool.get(gateway, GraphRuntimeOptions(snapshot=snapshot))
    GraphServiceRuntime(gateway=gateway, runtime=runtime).run_plugins(
        ("core_graph_metrics",),
        cfg=GraphMetricsStepConfig(snapshot=snapshot),
    )

    query_service = getattr(resource.backend, "service", None)
    query = getattr(query_service, "query", None)
    engine = getattr(query, "graph_engine", None)
    _expect(
        condition=engine is runtime.engine,
        detail="Serving backend should reuse pooled runtime engine",
    )
    _expect(
        condition=pool.get_calls > 0,
        detail="Runtime pool should be consulted during backend wiring",
    )


def test_serving_wiring_respects_supplied_runtime(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """build_backend_resource must not rebuild engines when runtime is provided."""
    gateway = provisioned_repo.gateway
    snapshot = _make_snapshot(provisioned_repo)
    cfg = ServingConfig(
        mode="local_db",
        repo_root=snapshot.repo_root,
        repo=snapshot.repo,
        commit=snapshot.commit,
        db_path=gateway.config.db_path,
    )
    runtime = build_graph_runtime(gateway, GraphRuntimeOptions(snapshot=snapshot))

    resource = build_backend_resource(
        cfg,
        gateway=gateway,
        options=BackendResourceOptions(graph_runtime=runtime),
    )

    query_service = getattr(resource.backend, "service", None)
    query = getattr(query_service, "query", None)
    engine = getattr(query, "graph_engine", None)
    _expect(
        condition=engine is runtime.engine,
        detail="Backend must respect the supplied runtime engine",
    )
