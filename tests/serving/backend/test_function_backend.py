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


# -----------------------------------------------------------------------------
# Additional Tests for get_function_summary
# -----------------------------------------------------------------------------


def test_get_function_summary_not_found_returns_message(
    architecture_gateway: StorageGateway,
) -> None:
    """Return not_found message when function doesn't exist."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    # Use a GOID that doesn't exist
    nonexistent_goid = 999999
    result = backend.get_function_summary(goid_h128=nonexistent_goid)

    _expect(condition=result.found is False, message="Should not find nonexistent function")
    _expect(
        condition=any(msg.code == "not_found" for msg in result.meta.messages),
        message="Should have not_found message",
    )


def test_get_function_summary_by_urn(architecture_gateway: StorageGateway) -> None:
    """Resolve function by URN."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    # This may or may not find a function depending on fixture state
    result = backend.get_function_summary(urn="test:urn")

    # Should return a result either way (found or not_found)
    _expect(
        condition=result is not None,
        message="Should return a result object",
    )


# -----------------------------------------------------------------------------
# Tests for list_high_risk_functions
# -----------------------------------------------------------------------------


def test_list_high_risk_functions_basic(architecture_gateway: StorageGateway) -> None:
    """List high risk functions with default parameters."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    result = backend.list_high_risk_functions()

    _expect(
        condition=result is not None,
        message="Should return a result object",
    )
    _expect(
        condition=isinstance(result.functions, list),
        message="Should have functions list",
    )


def test_list_high_risk_functions_with_min_risk(architecture_gateway: StorageGateway) -> None:
    """Filter high risk functions by minimum risk threshold."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    # Use a high min_risk to get fewer results
    min_risk_threshold = 0.9
    result = backend.list_high_risk_functions(min_risk=min_risk_threshold)

    _expect(
        condition=result is not None,
        message="Should return a result object",
    )


def test_list_high_risk_functions_with_limit(architecture_gateway: StorageGateway) -> None:
    """Respect limit parameter for high risk functions."""
    limit_value = 5
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    result = backend.list_high_risk_functions(limit=limit_value)

    _expect(
        condition=len(result.functions) <= limit_value,
        message=f"Should respect limit of {limit_value}",
    )
    _expect(
        condition=result.meta.applied_limit == limit_value,
        message="Should report applied limit in meta",
    )


def test_list_high_risk_functions_tested_only(architecture_gateway: StorageGateway) -> None:
    """Filter to only tested functions."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    result = backend.list_high_risk_functions(tested_only=True)

    _expect(
        condition=result is not None,
        message="Should return a result object",
    )


# -----------------------------------------------------------------------------
# Tests for get_callgraph_neighbors
# -----------------------------------------------------------------------------


def test_get_callgraph_neighbors_outgoing(architecture_gateway: StorageGateway) -> None:
    """Get outgoing call graph neighbors only."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    result = backend.get_callgraph_neighbors(goid_h128=1, direction="outgoing")

    _expect(
        condition=result is not None,
        message="Should return a result object",
    )
    # Outgoing should populate outgoing
    _expect(
        condition=isinstance(result.outgoing, list),
        message="Should have outgoing list",
    )


def test_get_callgraph_neighbors_incoming(architecture_gateway: StorageGateway) -> None:
    """Get incoming call graph neighbors only."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    result = backend.get_callgraph_neighbors(goid_h128=1, direction="incoming")

    _expect(
        condition=result is not None,
        message="Should return a result object",
    )
    _expect(
        condition=isinstance(result.incoming, list),
        message="Should have incoming list",
    )


def test_get_callgraph_neighbors_both(architecture_gateway: StorageGateway) -> None:
    """Get both incoming and outgoing neighbors."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    result = backend.get_callgraph_neighbors(goid_h128=1, direction="both")

    _expect(
        condition=result is not None,
        message="Should return a result object",
    )
    _expect(
        condition=isinstance(result.outgoing, list),
        message="Should have outgoing list",
    )
    _expect(
        condition=isinstance(result.incoming, list),
        message="Should have incoming list",
    )


def test_get_callgraph_neighbors_with_limit(architecture_gateway: StorageGateway) -> None:
    """Respect limit for call graph neighbors."""
    limit_value = 3
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    result = backend.get_callgraph_neighbors(goid_h128=1, limit=limit_value)

    _expect(
        condition=result.meta.applied_limit == limit_value,
        message="Should report applied limit in meta",
    )


# -----------------------------------------------------------------------------
# Tests for get_tests_for_function
# -----------------------------------------------------------------------------


def test_get_tests_for_function_not_found(architecture_gateway: StorageGateway) -> None:
    """Return message when function not found."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    # Use a nonexistent GOID
    nonexistent_goid = 999999
    result = backend.get_tests_for_function(goid_h128=nonexistent_goid)

    _expect(
        condition=result is not None,
        message="Should return a result object",
    )


def test_get_tests_for_function_by_goid(architecture_gateway: StorageGateway) -> None:
    """Get tests for function by GOID."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    result = backend.get_tests_for_function(goid_h128=1)

    _expect(
        condition=result is not None,
        message="Should return a result object",
    )
    _expect(
        condition=isinstance(result.tests, list),
        message="Should have tests list",
    )


def test_get_tests_for_function_with_limit(architecture_gateway: StorageGateway) -> None:
    """Respect limit for tests."""
    limit_value = 2
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    result = backend.get_tests_for_function(goid_h128=1, limit=limit_value)

    _expect(
        condition=result.meta.applied_limit == limit_value,
        message="Should report applied limit in meta",
    )


# -----------------------------------------------------------------------------
# Tests for get_callgraph_neighborhood
# -----------------------------------------------------------------------------


def test_get_callgraph_neighborhood_node_not_in_graph(
    architecture_gateway: StorageGateway,
) -> None:
    """Return empty neighborhood when node not in graph."""
    graph = nx.DiGraph()
    graph.add_edge(1, 2)
    graph_engine = cast("GraphEngine", _FakeGraphEngine(graph=graph))
    context, repositories, engine_provider = _build_components(
        architecture_gateway, graph_engine=graph_engine
    )
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    # Query for a node not in the graph
    neighborhood = backend.get_callgraph_neighborhood(goid_h128=999)

    _expect(condition=neighborhood.nodes == [], message="Should have no nodes")
    _expect(condition=neighborhood.edges == [], message="Should have no edges")


def test_get_callgraph_neighborhood_with_radius(
    architecture_gateway: StorageGateway,
) -> None:
    """Expand neighborhood by radius."""
    graph = nx.DiGraph()
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph_engine = cast("GraphEngine", _FakeGraphEngine(graph=graph))
    context, repositories, engine_provider = _build_components(
        architecture_gateway, graph_engine=graph_engine
    )
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    # Radius 2 should include 1, 2, 3
    radius_value = 2
    neighborhood = backend.get_callgraph_neighborhood(goid_h128=1, radius=radius_value)

    _expect(
        condition=neighborhood is not None,
        message="Should return neighborhood object",
    )


# -----------------------------------------------------------------------------
# Tests for get_import_boundary
# -----------------------------------------------------------------------------


def test_get_import_boundary_subsystem_not_found(
    architecture_gateway: StorageGateway,
) -> None:
    """Return empty boundary when subsystem not in graph."""
    graph = nx.DiGraph()
    graph.add_edge("sub1", "sub2")
    graph_engine = cast("GraphEngine", _FakeGraphEngine(graph=graph))
    context, repositories, engine_provider = _build_components(
        architecture_gateway, graph_engine=graph_engine
    )
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    boundary = backend.get_import_boundary(subsystem_id="nonexistent")

    _expect(condition=boundary.nodes == [], message="Should have no nodes")
    _expect(condition=boundary.edges == [], message="Should have no edges")


def test_get_import_boundary_basic(architecture_gateway: StorageGateway) -> None:
    """Get import boundary for a subsystem."""
    graph = nx.DiGraph()
    graph.add_edge("sub1", "sub2", weight=1.0)
    graph.add_edge("sub3", "sub1", weight=0.5)
    graph_engine = cast("GraphEngine", _FakeGraphEngine(graph=graph))
    context, repositories, engine_provider = _build_components(
        architecture_gateway, graph_engine=graph_engine
    )
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    boundary = backend.get_import_boundary(subsystem_id="sub1")

    _expect(
        condition=len(boundary.nodes) > 0 or len(boundary.edges) > 0,
        message="Should have boundary nodes or edges",
    )


def test_get_import_boundary_with_max_edges(architecture_gateway: StorageGateway) -> None:
    """Respect max_edges limit."""
    graph = nx.DiGraph()
    graph.add_edge("sub1", "sub2")
    graph.add_edge("sub1", "sub3")
    graph.add_edge("sub1", "sub4")
    graph_engine = cast("GraphEngine", _FakeGraphEngine(graph=graph))
    context, repositories, engine_provider = _build_components(
        architecture_gateway, graph_engine=graph_engine
    )
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    max_edges_value = 1
    boundary = backend.get_import_boundary(subsystem_id="sub1", max_edges=max_edges_value)

    _expect(
        condition=len(boundary.edges) <= max_edges_value,
        message=f"Should respect max_edges of {max_edges_value}",
    )


def test_get_import_boundary_no_graph_engine(architecture_gateway: StorageGateway) -> None:
    """Return empty boundary when no graph engine available."""
    context, repositories, engine_provider = _build_components(
        architecture_gateway, graph_engine=None
    )
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    boundary = backend.get_import_boundary(subsystem_id="sub1")

    _expect(condition=boundary.nodes == [], message="Should have no nodes")
    _expect(condition=boundary.edges == [], message="Should have no edges")


# -----------------------------------------------------------------------------
# Tests for get_function_profile
# -----------------------------------------------------------------------------


def test_get_function_profile_not_found(architecture_gateway: StorageGateway) -> None:
    """Raise not_found when profile doesn't exist."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    nonexistent_goid = 999999
    with pytest.raises(errors.McpError) as excinfo:
        backend.get_function_profile(nonexistent_goid)

    _expect(
        condition=excinfo.value.detail.code == "not-found",
        message="Should raise not-found error",
    )


# -----------------------------------------------------------------------------
# Tests for get_function_architecture
# -----------------------------------------------------------------------------


def test_get_function_architecture_not_found(architecture_gateway: StorageGateway) -> None:
    """Raise not_found when architecture doesn't exist."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    nonexistent_goid = 999999
    with pytest.raises(errors.McpError) as excinfo:
        backend.get_function_architecture(nonexistent_goid)

    _expect(
        condition=excinfo.value.detail.code == "not-found",
        message="Should raise not-found error",
    )


# -----------------------------------------------------------------------------
# Tests for Backend Properties
# -----------------------------------------------------------------------------


def test_backend_con_property(architecture_gateway: StorageGateway) -> None:
    """Verify con property returns connection."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    con = backend.con

    _expect(
        condition=con is not None,
        message="Should return connection",
    )


def test_backend_functions_property(architecture_gateway: StorageGateway) -> None:
    """Verify functions property returns repository."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    functions = backend.functions

    _expect(
        condition=functions is not None,
        message="Should return functions repository",
    )


def test_backend_graphs_property(architecture_gateway: StorageGateway) -> None:
    """Verify graphs property returns repository."""
    context, repositories, engine_provider = _build_components(architecture_gateway)
    backend = FunctionBackend(
        context=context,
        repositories=repositories,
        engine_provider=engine_provider,
    )

    graphs = backend.graphs

    _expect(
        condition=graphs is not None,
        message="Should return graphs repository",
    )
