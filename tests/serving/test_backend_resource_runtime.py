"""E2E wiring test for serving runtime/engine reuse."""

from __future__ import annotations

import pytest

from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    GraphRuntimePool,
    build_graph_runtime,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.serving.services.wiring import BackendResourceOptions, build_backend_resource
from tests._helpers.fixtures import ProvisionedGateway


def _expect(*, condition: bool, message: str) -> None:
    if condition:
        return
    pytest.fail(message)


def test_backend_resource_reuses_pooled_runtime(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Serving wiring should reuse the pooled runtime and its engine."""
    gateway = provisioned_repo.gateway
    cfg = ServingConfig(
        mode="local_db",
        repo_root=provisioned_repo.repo_root,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        db_path=gateway.config.db_path,
    )
    snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=cfg.repo_root)
    runtime = build_graph_runtime(
        gateway,
        GraphRuntimeOptions(snapshot=snapshot),
    )

    class _FixedPool(GraphRuntimePool):
        def __init__(self, runtime: GraphRuntime) -> None:
            super().__init__(max_size=1)
            self.runtime = runtime
            self.calls = 0

        def get(
            self,
            gateway: object,
            options: GraphRuntimeOptions,
            *,
            context: object | None = None,
        ) -> GraphRuntime:
            _ = gateway, options, context
            self.calls += 1
            return self.runtime

    fixed_pool = _FixedPool(runtime)

    resource = build_backend_resource(
        cfg,
        gateway=gateway,
        options=BackendResourceOptions(runtime_pool=fixed_pool),
    )

    query_service = getattr(resource.backend, "service", None)
    _expect(condition=query_service is not None, message="Backend should expose a service")
    query = getattr(query_service, "query", None)
    _expect(
        condition=query is not None,
        message="Query service should expose DuckDB query object",
    )
    engine = getattr(query, "graph_engine", None)
    if fixed_pool.calls == 0:
        pytest.fail("Runtime pool should be consulted during backend wiring")
    if engine is None or engine.repo != cfg.repo or engine.commit != cfg.commit:
        pytest.fail("Query engine should originate from pooled runtime with matching snapshot")


def test_backend_resource_reuse_across_builds(provisioned_repo: ProvisionedGateway) -> None:
    """Repeated backend builds with a shared pool should reuse the same engine."""
    gateway = provisioned_repo.gateway
    cfg = ServingConfig(
        mode="local_db",
        repo_root=provisioned_repo.repo_root,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        db_path=gateway.config.db_path,
    )
    snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=cfg.repo_root)
    runtime = build_graph_runtime(
        gateway,
        GraphRuntimeOptions(snapshot=snapshot),
    )

    class _FixedPool(GraphRuntimePool):
        def __init__(self, runtime: GraphRuntime) -> None:
            super().__init__(max_size=1)
            self.runtime = runtime

        def get(
            self,
            gateway: object,
            options: GraphRuntimeOptions,
            *,
            context: object | None = None,
        ) -> GraphRuntime:
            _ = gateway, options, context
            return self.runtime

    pool = _FixedPool(runtime)
    res1 = build_backend_resource(
        cfg,
        gateway=gateway,
        options=BackendResourceOptions(runtime_pool=pool),
    )
    res2 = build_backend_resource(
        cfg,
        gateway=gateway,
        options=BackendResourceOptions(runtime_pool=pool),
    )

    query_service1 = getattr(res1.backend, "service", None)
    query1 = getattr(query_service1, "query", None) if query_service1 is not None else None
    query_service2 = getattr(res2.backend, "service", None)
    query2 = getattr(query_service2, "query", None) if query_service2 is not None else None
    _expect(
        condition=query1 is not None and query2 is not None,
        message="Both backends should expose query services",
    )
    engine1 = query1.graph_engine if query1 is not None else None
    engine2 = query2.graph_engine if query2 is not None else None
    if engine1 is not runtime.engine or engine2 is not runtime.engine:
        pytest.fail("All backend builds should reuse the pooled runtime engine")
