"""Tests demonstrating MockGraphRuntime usage in graph plugin scenarios.

This module provides test patterns and examples for using MockGraphRuntime
when testing graph-related functionality. These tests serve both as
coverage and as documentation for how to use the helper effectively.

Key patterns demonstrated:
1. Testing GraphProvider with mock runtimes
2. Testing graph computation with controlled inputs
3. Testing graph validation with specific graph shapes
4. Testing plugin execution with pre-configured graphs
"""

from __future__ import annotations

from typing import Final

import networkx as nx
import pytest

from codeintel.analytics.resources.graphs import GraphProvider
from codeintel.config.primitives import GraphBackendConfig
from tests._helpers.assertions import (
    expect_equal,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.fakes.graph_runtime import (
    GraphRuntimeDouble as MockGraphRuntime,
)
from tests._helpers.fakes.graph_runtime import (
    create_mock_runtime_all_graphs,
    create_mock_runtime_with_call_graph,
    create_mock_runtime_with_import_graph,
    create_mock_runtime_with_standard_graphs,
)
from tests._helpers.graphs import (
    call_chain_graph,
    call_star_graph,
    import_cycle_graph,
    standard_graph_fixtures,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_CHAIN_NODES: Final = 4
EXPECTED_STAR_EDGES: Final = 3
EXPECTED_CYCLE_SIZE: Final = 3


# ===========================================================================
# GraphProvider with MockGraphRuntime Tests
# ===========================================================================


class TestGraphProviderWithMockRuntime:
    """Test GraphProvider behavior using MockGraphRuntime."""

    @staticmethod
    def test_provider_loads_call_graph_from_mock(
        mock_runtime_with_call_graph: MockGraphRuntime,
    ) -> None:
        """GraphProvider correctly loads call graph from mock runtime."""
        provider = GraphProvider.from_runtime(mock_runtime_with_call_graph)
        resources = provider.get()

        call_graph = expect_is_not_none(
            resources.call_graph, message="Expected call graph to be set"
        )
        expect_true(call_graph.number_of_edges() > 0)

    @staticmethod
    def test_provider_loads_import_graph_from_mock(
        mock_runtime_with_import_graph: MockGraphRuntime,
    ) -> None:
        """GraphProvider correctly loads import graph from mock runtime."""
        provider = GraphProvider.from_runtime(mock_runtime_with_import_graph)
        resources = provider.get()

        import_graph = expect_is_not_none(
            resources.import_graph, message="Expected import graph to be set"
        )
        expect_true(import_graph.number_of_edges() > 0)

    @staticmethod
    def test_provider_loads_all_graphs_from_mock(
        mock_runtime_all_graphs: MockGraphRuntime,
    ) -> None:
        """GraphProvider loads all graph types from comprehensive mock."""
        provider = GraphProvider.from_runtime(mock_runtime_all_graphs)
        resources = provider.get()

        # All graphs should be available
        expect_is_not_none(resources.call_graph)
        expect_is_not_none(resources.import_graph)
        expect_is_not_none(resources.symbol_module_graph)
        expect_is_not_none(resources.symbol_function_graph)
        expect_is_not_none(resources.config_module_bipartite)
        expect_is_not_none(resources.test_function_bipartite)
        expect_is_not_none(resources.cfg_graph)

    @staticmethod
    def test_provider_empty_mock_returns_none_graphs(
        mock_graph_runtime: MockGraphRuntime,
    ) -> None:
        """GraphProvider returns None for graphs not set in mock."""
        provider = GraphProvider.from_runtime(mock_graph_runtime)
        resources = provider.get()

        expect_is_none(resources.call_graph)
        expect_is_none(resources.import_graph)

    @staticmethod
    def test_provider_backend_from_mock() -> None:
        """GraphProvider exposes backend config from mock runtime."""
        backend = GraphBackendConfig(use_gpu=True)
        mock = MockGraphRuntime(backend=backend, use_gpu=True)
        provider = GraphProvider.from_runtime(mock)

        expect_true(provider.backend is backend)
        expect_true(provider.use_gpu)


# ===========================================================================
# Custom Graph Shapes with MockGraphRuntime Tests
# ===========================================================================


class TestCustomGraphShapes:
    """Test MockGraphRuntime with various graph topologies."""

    @staticmethod
    def test_chain_graph_topology() -> None:
        """MockGraphRuntime works with chain graph topology."""
        call_g = call_chain_graph(EXPECTED_CHAIN_NODES)
        mock = MockGraphRuntime(call_graph=call_g)
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        call_graph = expect_is_not_none(
            resources.call_graph, message="Expected call graph to be set"
        )
        expect_equal(call_graph.number_of_nodes(), EXPECTED_CHAIN_NODES)

    @staticmethod
    def test_star_graph_topology() -> None:
        """MockGraphRuntime works with star graph topology."""
        call_g = call_star_graph(EXPECTED_STAR_EDGES, inward=True)
        mock = MockGraphRuntime(call_graph=call_g)
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        call_graph = expect_is_not_none(
            resources.call_graph, message="Expected call graph to be set"
        )
        expect_equal(call_graph.number_of_edges(), EXPECTED_STAR_EDGES)

    @staticmethod
    def test_cyclic_graph_topology() -> None:
        """MockGraphRuntime works with cyclic graph topology."""
        import_g = import_cycle_graph(EXPECTED_CYCLE_SIZE)
        mock = MockGraphRuntime(import_graph=import_g)
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        import_graph = expect_is_not_none(
            resources.import_graph, message="Expected import graph to be set"
        )
        # Cyclic graph has same number of edges as nodes
        expect_equal(import_graph.number_of_edges(), EXPECTED_CYCLE_SIZE)

    @staticmethod
    def test_diamond_graph_topology() -> None:
        """MockGraphRuntime works with diamond graph topology."""
        fixtures = standard_graph_fixtures(chain_length=EXPECTED_CHAIN_NODES)
        mock = create_mock_runtime_with_standard_graphs(fixtures)
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        call_graph = expect_is_not_none(
            resources.call_graph, message="Expected call graph to be set"
        )
        expect_equal(call_graph.number_of_edges(), EXPECTED_CHAIN_NODES - 1)


# ===========================================================================
# Mixed Graph Types Tests
# ===========================================================================


class TestMixedGraphTypes:
    """Test MockGraphRuntime with mixed graph types."""

    @staticmethod
    def test_directed_and_undirected_together() -> None:
        """MockGraphRuntime handles both directed and undirected graphs."""
        # Directed graphs
        call_g = nx.DiGraph([("a", "b"), ("b", "c")])
        import_g = nx.DiGraph([("mod1", "mod2")])

        # Undirected graphs
        symbol_mod_g = nx.Graph([("sym1", "mod1"), ("sym2", "mod2")])

        mock = MockGraphRuntime(
            call_graph=call_g,
            import_graph=import_g,
            symbol_module_graph=symbol_mod_g,
        )
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        # Directed graphs
        call_graph = expect_is_not_none(
            resources.call_graph, message="Expected call graph to be set"
        )
        import_graph = expect_is_not_none(
            resources.import_graph, message="Expected import graph to be set"
        )
        expect_true(isinstance(call_graph, nx.DiGraph))
        expect_true(isinstance(import_graph, nx.DiGraph))

        # Undirected graph
        symbol_module_graph = expect_is_not_none(
            resources.symbol_module_graph, message="Expected symbol_module_graph to be set"
        )
        expect_true(isinstance(symbol_module_graph, nx.Graph))

    @staticmethod
    def test_cfg_graph_structure() -> None:
        """MockGraphRuntime preserves CFG graph structure."""
        # Create a simple CFG with entry/exit blocks
        cfg = nx.DiGraph()
        cfg.add_edges_from(
            [
                ("entry", "block1"),
                ("block1", "block2"),
                ("block1", "block3"),  # Branch
                ("block2", "exit"),
                ("block3", "exit"),
            ]
        )

        mock = MockGraphRuntime(cfg_graph=cfg)
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        cfg_graph = expect_is_not_none(resources.cfg_graph, message="Expected cfg_graph to be set")
        expect_true("entry" in cfg_graph.nodes)
        expect_true("exit" in cfg_graph.nodes)


# ===========================================================================
# Graph Resource Caching Tests
# ===========================================================================


class TestGraphResourceCaching:
    """Test that GraphProvider caches resources from MockGraphRuntime."""

    @staticmethod
    def test_resources_cached_on_get(mock_runtime_with_call_graph: MockGraphRuntime) -> None:
        """GraphProvider caches resources after first get() call."""
        provider = GraphProvider.from_runtime(mock_runtime_with_call_graph)

        resources1 = provider.get()
        resources2 = provider.get()

        # Should return same cached instance
        expect_true(resources1 is resources2)

    @staticmethod
    def test_invalidation_clears_cache(
        mock_runtime_with_call_graph: MockGraphRuntime,
    ) -> None:
        """Invalidation clears the cached resources."""
        provider = GraphProvider.from_runtime(mock_runtime_with_call_graph)

        resources1 = provider.get()
        provider.invalidate()
        resources2 = provider.get()

        # Should get fresh resources after invalidation
        # Note: for mocks without state changes, content is same but instance may differ
        expect_true(resources1 is not resources2)


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestMockRuntimeFactories:
    """Test the mock runtime factory functions."""

    @staticmethod
    def test_create_mock_runtime_with_call_graph_defaults() -> None:
        """Factory creates mock with default call graph edges."""
        mock = create_mock_runtime_with_call_graph()

        call_graph = mock.call_graph
        if call_graph is None:
            pytest.fail("Expected call graph to be set")
        expect_true(call_graph.number_of_edges() > 0)
        expect_true("func_a" in call_graph.nodes)
        expect_true("func_b" in call_graph.nodes)

    @staticmethod
    def test_create_mock_runtime_with_call_graph_custom() -> None:
        """Factory creates mock with custom call graph edges."""
        custom_edges = [("main", "helper"), ("helper", "util")]
        mock = create_mock_runtime_with_call_graph(custom_edges)

        call_graph = mock.call_graph
        if call_graph is None:
            pytest.fail("Expected call graph to be set")
        expect_true("main" in call_graph.nodes)
        expect_true("helper" in call_graph.nodes)
        expect_true("util" in call_graph.nodes)

    @staticmethod
    def test_create_mock_runtime_with_import_graph_defaults() -> None:
        """Factory creates mock with default import graph edges."""
        mock = create_mock_runtime_with_import_graph()

        import_graph = mock.import_graph
        if import_graph is None:
            pytest.fail("Expected import graph to be set")
        expect_true(import_graph.number_of_edges() > 0)

    @staticmethod
    def test_create_mock_runtime_all_graphs_coverage() -> None:
        """Factory creates mock with all graph types populated."""
        mock = create_mock_runtime_all_graphs()

        # All graph properties should be non-None
        expect_true(mock.call_graph is not None)
        expect_true(mock.import_graph is not None)
        expect_true(mock.symbol_module_graph is not None)
        expect_true(mock.symbol_function_graph is not None)
        expect_true(mock.config_module_bipartite is not None)
        expect_true(mock.test_function_bipartite is not None)
        expect_true(mock.cfg_graph is not None)


# ===========================================================================
# Ensure Methods Tests
# ===========================================================================


class TestEnsureMethods:
    """Test the ensure_* methods on MockGraphRuntime."""

    @staticmethod
    def test_ensure_call_graph_returns_graph() -> None:
        """ensure_call_graph returns the call graph."""
        call_g = nx.DiGraph([("a", "b")])
        mock = MockGraphRuntime(call_graph=call_g, copy_graphs=False)

        result = mock.ensure_call_graph()

        expect_true(result is call_g)

    @staticmethod
    def test_ensure_import_graph_returns_graph() -> None:
        """ensure_import_graph returns the import graph."""
        import_g = nx.DiGraph([("mod1", "mod2")])
        mock = MockGraphRuntime(import_graph=import_g, copy_graphs=False)

        result = mock.ensure_import_graph()

        expect_true(result is import_g)

    @staticmethod
    def test_ensure_returns_none_when_not_set() -> None:
        """ensure_* methods return None when graph not set."""
        mock = MockGraphRuntime()

        expect_is_none(mock.ensure_call_graph())
        expect_is_none(mock.ensure_import_graph())
        expect_is_none(mock.ensure_symbol_module_graph())
        expect_is_none(mock.ensure_symbol_function_graph())
        expect_is_none(mock.ensure_config_module_bipartite())
        expect_is_none(mock.ensure_test_function_bipartite())
        expect_is_none(mock.ensure_cfg_graph())
