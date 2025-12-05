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

from codeintel.analytics.resources.graphs import GraphProvider
from codeintel.config.primitives import GraphBackendConfig
from tests._helpers.fakes.graph_runtimes import (
    MockGraphRuntime,
    create_mock_runtime_all_graphs,
    create_mock_runtime_with_call_graph,
    create_mock_runtime_with_import_graph,
)
from tests._helpers.fakes.networkx_graphs import (
    chain_graph,
    cyclic_graph,
    diamond_graph,
    star_graph,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_CHAIN_NODES: Final = 4
EXPECTED_STAR_EDGES: Final = 3
EXPECTED_CYCLE_SIZE: Final = 3
EXPECTED_DIAMOND_EDGES: Final = 4


# ===========================================================================
# GraphProvider with MockGraphRuntime Tests
# ===========================================================================


class TestGraphProviderWithMockRuntime:
    """Test GraphProvider behavior using MockGraphRuntime."""

    def test_provider_loads_call_graph_from_mock(
        self, mock_runtime_with_call_graph: MockGraphRuntime
    ) -> None:
        """GraphProvider correctly loads call graph from mock runtime."""
        provider = GraphProvider.from_runtime(mock_runtime_with_call_graph)
        resources = provider.get()

        assert resources.call_graph is not None
        assert resources.call_graph.number_of_edges() > 0

    def test_provider_loads_import_graph_from_mock(
        self, mock_runtime_with_import_graph: MockGraphRuntime
    ) -> None:
        """GraphProvider correctly loads import graph from mock runtime."""
        provider = GraphProvider.from_runtime(mock_runtime_with_import_graph)
        resources = provider.get()

        assert resources.import_graph is not None
        assert resources.import_graph.number_of_edges() > 0

    def test_provider_loads_all_graphs_from_mock(
        self, mock_runtime_all_graphs: MockGraphRuntime
    ) -> None:
        """GraphProvider loads all graph types from comprehensive mock."""
        provider = GraphProvider.from_runtime(mock_runtime_all_graphs)
        resources = provider.get()

        # All graphs should be available
        assert resources.call_graph is not None
        assert resources.import_graph is not None
        assert resources.symbol_module_graph is not None
        assert resources.symbol_function_graph is not None
        assert resources.config_module_bipartite is not None
        assert resources.test_function_bipartite is not None
        assert resources.cfg_graph is not None

    def test_provider_empty_mock_returns_none_graphs(
        self, mock_graph_runtime: MockGraphRuntime
    ) -> None:
        """GraphProvider returns None for graphs not set in mock."""
        provider = GraphProvider.from_runtime(mock_graph_runtime)
        resources = provider.get()

        assert resources.call_graph is None
        assert resources.import_graph is None

    def test_provider_backend_from_mock(self) -> None:
        """GraphProvider exposes backend config from mock runtime."""
        backend = GraphBackendConfig(use_gpu=True)
        mock = MockGraphRuntime(backend=backend, use_gpu=True)
        provider = GraphProvider.from_runtime(mock)

        assert provider.backend is backend
        assert provider.use_gpu is True


# ===========================================================================
# Custom Graph Shapes with MockGraphRuntime Tests
# ===========================================================================


class TestCustomGraphShapes:
    """Test MockGraphRuntime with various graph topologies."""

    def test_chain_graph_topology(self) -> None:
        """MockGraphRuntime works with chain graph topology."""
        call_g = chain_graph(EXPECTED_CHAIN_NODES)
        mock = MockGraphRuntime(call_graph=call_g)
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        assert resources.call_graph is not None
        assert resources.call_graph.number_of_nodes() == EXPECTED_CHAIN_NODES

    def test_star_graph_topology(self) -> None:
        """MockGraphRuntime works with star graph topology."""
        call_g = star_graph(EXPECTED_STAR_EDGES, inward=True)
        mock = MockGraphRuntime(call_graph=call_g)
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        assert resources.call_graph is not None
        assert resources.call_graph.number_of_edges() == EXPECTED_STAR_EDGES

    def test_cyclic_graph_topology(self) -> None:
        """MockGraphRuntime works with cyclic graph topology."""
        import_g = cyclic_graph(EXPECTED_CYCLE_SIZE)
        mock = MockGraphRuntime(import_graph=import_g)
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        assert resources.import_graph is not None
        # Cyclic graph has same number of edges as nodes
        assert resources.import_graph.number_of_edges() == EXPECTED_CYCLE_SIZE

    def test_diamond_graph_topology(self) -> None:
        """MockGraphRuntime works with diamond graph topology."""
        call_g = diamond_graph()
        mock = MockGraphRuntime(call_graph=call_g)
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        assert resources.call_graph is not None
        assert resources.call_graph.number_of_edges() == EXPECTED_DIAMOND_EDGES


# ===========================================================================
# Mixed Graph Types Tests
# ===========================================================================


class TestMixedGraphTypes:
    """Test MockGraphRuntime with mixed graph types."""

    def test_directed_and_undirected_together(self) -> None:
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
        assert isinstance(resources.call_graph, nx.DiGraph)
        assert isinstance(resources.import_graph, nx.DiGraph)

        # Undirected graph
        assert isinstance(resources.symbol_module_graph, nx.Graph)

    def test_cfg_graph_structure(self) -> None:
        """MockGraphRuntime preserves CFG graph structure."""
        # Create a simple CFG with entry/exit blocks
        cfg = nx.DiGraph()
        cfg.add_edges_from([
            ("entry", "block1"),
            ("block1", "block2"),
            ("block1", "block3"),  # Branch
            ("block2", "exit"),
            ("block3", "exit"),
        ])

        mock = MockGraphRuntime(cfg_graph=cfg)
        provider = GraphProvider.from_runtime(mock)
        resources = provider.get()

        assert resources.cfg_graph is not None
        assert "entry" in resources.cfg_graph.nodes
        assert "exit" in resources.cfg_graph.nodes


# ===========================================================================
# Graph Resource Caching Tests
# ===========================================================================


class TestGraphResourceCaching:
    """Test that GraphProvider caches resources from MockGraphRuntime."""

    def test_resources_cached_on_get(
        self, mock_runtime_with_call_graph: MockGraphRuntime
    ) -> None:
        """GraphProvider caches resources after first get() call."""
        provider = GraphProvider.from_runtime(mock_runtime_with_call_graph)

        resources1 = provider.get()
        resources2 = provider.get()

        # Should return same cached instance
        assert resources1 is resources2

    def test_invalidation_clears_cache(
        self, mock_runtime_with_call_graph: MockGraphRuntime
    ) -> None:
        """Invalidation clears the cached resources."""
        provider = GraphProvider.from_runtime(mock_runtime_with_call_graph)

        resources1 = provider.get()
        provider.invalidate()
        resources2 = provider.get()

        # Should get fresh resources after invalidation
        # Note: for mocks without state changes, content is same but instance may differ
        assert resources1 is not resources2


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestMockRuntimeFactories:
    """Test the mock runtime factory functions."""

    def test_create_mock_runtime_with_call_graph_defaults(self) -> None:
        """Factory creates mock with default call graph edges."""
        mock = create_mock_runtime_with_call_graph()

        assert mock.call_graph is not None
        assert mock.call_graph.number_of_edges() > 0
        assert "func_a" in mock.call_graph.nodes
        assert "func_b" in mock.call_graph.nodes

    def test_create_mock_runtime_with_call_graph_custom(self) -> None:
        """Factory creates mock with custom call graph edges."""
        custom_edges = [("main", "helper"), ("helper", "util")]
        mock = create_mock_runtime_with_call_graph(custom_edges)

        assert mock.call_graph is not None
        assert "main" in mock.call_graph.nodes
        assert "helper" in mock.call_graph.nodes
        assert "util" in mock.call_graph.nodes

    def test_create_mock_runtime_with_import_graph_defaults(self) -> None:
        """Factory creates mock with default import graph edges."""
        mock = create_mock_runtime_with_import_graph()

        assert mock.import_graph is not None
        assert mock.import_graph.number_of_edges() > 0

    def test_create_mock_runtime_all_graphs_coverage(self) -> None:
        """Factory creates mock with all graph types populated."""
        mock = create_mock_runtime_all_graphs()

        # All graph properties should be non-None
        assert mock.call_graph is not None
        assert mock.import_graph is not None
        assert mock.symbol_module_graph is not None
        assert mock.symbol_function_graph is not None
        assert mock.config_module_bipartite is not None
        assert mock.test_function_bipartite is not None
        assert mock.cfg_graph is not None


# ===========================================================================
# Ensure Methods Tests
# ===========================================================================


class TestEnsureMethods:
    """Test the ensure_* methods on MockGraphRuntime."""

    def test_ensure_call_graph_returns_graph(self) -> None:
        """ensure_call_graph returns the call graph."""
        call_g = nx.DiGraph([("a", "b")])
        mock = MockGraphRuntime(call_graph=call_g)

        result = mock.ensure_call_graph()

        assert result is call_g

    def test_ensure_import_graph_returns_graph(self) -> None:
        """ensure_import_graph returns the import graph."""
        import_g = nx.DiGraph([("mod1", "mod2")])
        mock = MockGraphRuntime(import_graph=import_g)

        result = mock.ensure_import_graph()

        assert result is import_g

    def test_ensure_returns_none_when_not_set(self) -> None:
        """ensure_* methods return None when graph not set."""
        mock = MockGraphRuntime()

        assert mock.ensure_call_graph() is None
        assert mock.ensure_import_graph() is None
        assert mock.ensure_symbol_module_graph() is None
        assert mock.ensure_symbol_function_graph() is None
        assert mock.ensure_config_module_bipartite() is None
        assert mock.ensure_test_function_bipartite() is None
        assert mock.ensure_cfg_graph() is None
