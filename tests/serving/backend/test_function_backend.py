"""Tests for FunctionBackend behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import networkx as nx
import pytest

from codeintel.graphs.engine import GraphEngine
from codeintel.serving.backend import BackendContext, BackendLimits, DuckDBRepositories
from codeintel.serving.backend.core import GraphEngineProvider
from codeintel.serving.backend.function_backend import FunctionBackend
from codeintel.serving.mcp import errors
from codeintel.storage.gateway import StorageGateway


def _expect(*, condition: bool, message: str) -> None:
    """Fail the test when a condition is not met."""
    if not condition:
        pytest.fail(message)


@dataclass
class _FakeGraphEngine:
    """Minimal graph engine stub for neighborhood queries."""

    graph: nx.DiGraph

    def call_graph(self) -> nx.DiGraph:
        return self.graph

    def import_graph(self) -> nx.DiGraph:
        return self.graph


def _build_components(
    gateway: StorageGateway,
    *,
    limits: BackendLimits | None = None,
    graph_engine: GraphEngine | None = None,
) -> tuple[BackendContext, DuckDBRepositories, GraphEngineProvider]:
    repo = gateway.config.repo or "demo/repo"
    commit = gateway.config.commit or "deadbeef"
    context = BackendContext(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=limits or BackendLimits(),
        graph_engine=graph_engine,
    )
    repositories = DuckDBRepositories(gateway, context.repo, context.commit)
    engine_provider = GraphEngineProvider(context=context, graph_engine=graph_engine)
    return context, repositories, engine_provider


def test_get_function_summary_requires_identifier(architecture_gateway: StorageGateway) -> None:
    """Verify missing identifiers raise an invalid-argument problem."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    with pytest.raises(errors.McpError) as excinfo:
        backend.get_function_summary()

    _expect(
        condition=excinfo.value.detail.code == "invalid-argument",
        message="Missing identifiers should raise invalid-argument",
    )


def test_get_function_summary_found(architecture_gateway: StorageGateway) -> None:
    """Return function summary for seeded GOID."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    result = backend.get_function_summary(goid_h128=1)

    _expect(condition=result.found is True, message="Function should be found")
    if result.summary is None:
        pytest.fail("Function summary should not be None for seeded function")
    _expect(
        condition=result.summary.get("function_goid_h128") == 1,
        message="Seeded GOID should be returned in summary",
    )


def test_get_callgraph_neighborhood_truncates(
    architecture_gateway: StorageGateway,
) -> None:
    """Truncate neighborhoods when max_nodes is enforced."""
    graph = nx.DiGraph()
    graph.add_edge(1, 1)
    graph_engine = cast("GraphEngine", _FakeGraphEngine(graph=graph))
    context, repositories, engine_provider = _build_components(
        architecture_gateway, graph_engine=graph_engine
    )
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    neighborhood = backend.get_callgraph_neighborhood(goid_h128=1, max_nodes=0)

    _expect(condition=neighborhood.meta.truncated is True, message="Neighborhood should truncate")
    _expect(condition=neighborhood.nodes == [], message="No nodes should remain after truncation")
    _expect(condition=neighborhood.edges == [], message="No edges should remain after truncation")
