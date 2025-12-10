"""Shared fixtures for CLI handler tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest

from codeintel.analytics.runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.cli.handlers.context import HandlerContext
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageGateway
from tests._helpers.configs import ProvisionedGateway
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.serving_contexts import (
    ProvisionedServiceContext,
    build_provisioned_service_context,
)
from tests.serving.mcp.conftest import McpBackendComponents

type HandlerContextBuilder = Callable[
    [ProvisionedServiceContext, str, dict[str, object]],
    HandlerContext,
]


class FakeGraphEngine:
    """Minimal GraphEngine implementation for handler tests."""

    def __init__(
        self,
        snapshot: SnapshotRef,
        gateway: StorageGateway | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.gateway: StorageGateway = gateway or MagicMock(spec=StorageGateway)

    @property
    def use_gpu(self) -> bool:
        """Indicate GPU is not used for the fake engine."""
        return False

    def call_graph(self) -> nx.DiGraph:
        """Return an empty call graph.

        Returns
        -------
        nx.DiGraph
            Empty directed graph placeholder.
        """
        return self.load_call_graph()

    @staticmethod
    def load_call_graph() -> nx.DiGraph:
        """Return an empty call graph.

        Returns
        -------
        nx.DiGraph
            Empty directed graph placeholder.
        """
        return nx.DiGraph()

    def import_graph(self) -> nx.DiGraph:
        """Return an empty import graph.

        Returns
        -------
        nx.DiGraph
            Empty directed graph placeholder.
        """
        return self.load_import_graph()

    @staticmethod
    def load_import_graph() -> nx.DiGraph:
        """Return an empty import graph.

        Returns
        -------
        nx.DiGraph
            Empty directed graph placeholder.
        """
        return nx.DiGraph()

    def symbol_module_graph(self) -> nx.Graph:
        """Return an empty symbol-module graph.

        Returns
        -------
        nx.Graph
            Empty undirected graph placeholder.
        """
        return self.load_symbol_module_graph()

    @staticmethod
    def load_symbol_module_graph() -> nx.Graph:
        """Return an empty symbol-module graph.

        Returns
        -------
        nx.Graph
            Empty undirected graph placeholder.
        """
        return nx.Graph()

    def symbol_function_graph(self) -> nx.Graph:
        """Return an empty symbol-function graph.

        Returns
        -------
        nx.Graph
            Empty undirected graph placeholder.
        """
        return self.load_symbol_function_graph()

    @staticmethod
    def load_symbol_function_graph() -> nx.Graph:
        """Return an empty symbol-function graph.

        Returns
        -------
        nx.Graph
            Empty undirected graph placeholder.
        """
        return nx.Graph()

    def config_module_bipartite(self) -> nx.Graph:
        """Return an empty config-module bipartite graph.

        Returns
        -------
        nx.Graph
            Empty undirected graph placeholder.
        """
        return self.load_config_module_bipartite()

    @staticmethod
    def load_config_module_bipartite() -> nx.Graph:
        """Return an empty config-module bipartite graph.

        Returns
        -------
        nx.Graph
            Empty undirected graph placeholder.
        """
        return nx.Graph()

    def test_function_bipartite(self) -> nx.Graph:
        """Return an empty test-function bipartite graph.

        Returns
        -------
        nx.Graph
            Empty undirected graph placeholder.
        """
        return self.load_test_function_bipartite()

    @staticmethod
    def load_test_function_bipartite() -> nx.Graph:
        """Return an empty test-function bipartite graph.

        Returns
        -------
        nx.Graph
            Empty undirected graph placeholder.
        """
        return nx.Graph()


class FakeGraphRuntime(GraphRuntime):
    """Minimal graph runtime stand-in for handler tests."""

    def __init__(
        self,
        snapshot: SnapshotRef,
        gateway: StorageGateway | None = None,
        backend: GraphBackendConfig | None = None,
    ) -> None:
        engine = FakeGraphEngine(snapshot=snapshot, gateway=gateway)
        options = GraphRuntimeOptions(snapshot=snapshot, backend=backend)
        super().__init__(options=options, engine=engine)


@pytest.fixture
def handler_service_context(
    provisioned_repo: ProvisionedGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> ProvisionedServiceContext:
    """Provisioned LocalQueryService/Backend for handler tests.

    Returns
    -------
    ProvisionedServiceContext
        Context built from the ingested repository snapshot.
    """
    return build_provisioned_service_context(
        mcp_backend_factory,
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
    )


@pytest.fixture
def architecture_service_context(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> ProvisionedServiceContext:
    """Provisioned context seeded with architecture data.

    Returns
    -------
    ProvisionedServiceContext
        Context backed by architecture gateway seeds.
    """
    return build_provisioned_service_context(
        mcp_backend_factory,
        gateway=architecture_gateway,
        snapshot=(DEFAULT_REPO, DEFAULT_COMMIT),
    )


@pytest.fixture
def handler_context_builder() -> HandlerContextBuilder:
    """Build a HandlerContext wired to a provisioned service backend.

    Returns
    -------
    HandlerContextBuilder
        Callable that constructs a HandlerContext bound to the given service context.
    """

    def _build(
        service_ctx: ProvisionedServiceContext,
        operation_id: str,
        params: dict[str, object],
    ) -> HandlerContext:
        gateway_config = getattr(service_ctx.gateway, "config", None)
        repo_root = getattr(gateway_config, "repo_root", Path.cwd())
        db_path = getattr(gateway_config, "db_path", None)
        serving = ServingConfig(
            mode="local_db",
            repo_root=repo_root,
            repo=service_ctx.repo,
            commit=service_ctx.commit,
            db_path=db_path,
            default_limit=service_ctx.limits.default_limit,
            max_rows_per_call=service_ctx.limits.max_rows_per_call,
        )

        runtime = MagicMock()
        runtime.serving = serving
        runtime.paths = MagicMock()
        runtime.paths.db_path = db_path
        runtime.repo = service_ctx.repo
        runtime.commit = service_ctx.commit

        snapshot = SnapshotRef(
            repo=service_ctx.repo, commit=service_ctx.commit, repo_root=repo_root
        )
        graph_runtime = FakeGraphRuntime(
            snapshot=snapshot,
            gateway=service_ctx.gateway,
            backend=GraphBackendConfig(),
        )
        return HandlerContext(
            config=MagicMock(),
            operation_id=operation_id,
            _params=params,
            _runtime=runtime,
            _gateway=service_ctx.gateway,
            _graph_runtime=graph_runtime,
        )

    return _build
