"""Tests for graph resource provider.

This module tests:
- GraphResources dataclass
- GraphProvider lazy loading behavior
- SingleGraphProvider for individual graphs
"""

from __future__ import annotations

import networkx as nx
import pytest

from codeintel.analytics.resources.graphs import (
    GraphProvider,
    GraphResources,
    SingleGraphProvider,
)
from codeintel.analytics.resources.protocol import ResourceNotLoadedError
from codeintel.config.primitives import GraphBackendConfig
from tests._helpers.fakes import MockGraphRuntime

# ============================================================================
# GraphResources Tests
# ============================================================================


def test_graph_resources_empty() -> None:
    """Empty resources have all None fields."""
    resources = GraphResources()

    assert resources.call_graph is None
    assert resources.import_graph is None
    assert resources.symbol_module_graph is None
    assert resources.symbol_function_graph is None
    assert resources.config_module_bipartite is None
    assert resources.test_function_bipartite is None
    assert resources.cfg_graph is None


def test_graph_resources_with_graphs() -> None:
    """Resources can hold graph instances."""
    call_g = nx.DiGraph()
    call_g.add_edge("A", "B")
    import_g = nx.DiGraph()
    import_g.add_edge("mod_a", "mod_b")

    resources = GraphResources(
        call_graph=call_g,
        import_graph=import_g,
    )

    assert resources.call_graph is call_g
    assert resources.import_graph is import_g
    # Access through typed local variables to verify methods work
    assert call_g.number_of_edges() == 1
    assert import_g.number_of_edges() == 1


# ============================================================================
# GraphProvider Tests
# ============================================================================


def test_provider_from_runtime() -> None:
    """Provider can be created from runtime."""
    call_g = nx.DiGraph()
    call_g.add_edge("f1", "f2")
    mock_runtime = MockGraphRuntime(call_graph=call_g)

    provider = GraphProvider.from_runtime(mock_runtime)

    assert provider.resource_name == "GraphResources"
    assert not provider.is_loaded


def test_provider_loads_on_get() -> None:
    """Provider loads resources on first get()."""
    call_g = nx.DiGraph()
    call_g.add_edge("f1", "f2")
    import_g = nx.DiGraph()
    import_g.add_edge("m1", "m2")
    mock_runtime = MockGraphRuntime(
        call_graph=call_g,
        import_graph=import_g,
    )

    provider = GraphProvider.from_runtime(mock_runtime)
    resources = provider.get()

    assert provider.is_loaded
    call_result = resources.call_graph
    import_result = resources.import_graph
    assert call_result is not None
    assert import_result is not None
    assert call_result.number_of_edges() == 1
    assert import_result.number_of_edges() == 1


def test_provider_caches_resources() -> None:
    """Provider caches resources after first load."""
    mock_runtime = MockGraphRuntime(call_graph=nx.DiGraph())
    provider = GraphProvider.from_runtime(mock_runtime)

    resources1 = provider.get()
    resources2 = provider.get()

    assert resources1 is resources2


def test_provider_invalidate_clears_cache() -> None:
    """Invalidate clears cached resources."""
    mock_runtime = MockGraphRuntime(call_graph=nx.DiGraph())
    provider = GraphProvider.from_runtime(mock_runtime)

    _ = provider.get()
    assert provider.is_loaded

    provider.invalidate()
    assert not provider.is_loaded


def test_provider_call_graph_property() -> None:
    """Call graph property loads and returns graph."""
    call_g = nx.DiGraph()
    call_g.add_edge("f1", "f2")
    mock_runtime = MockGraphRuntime(call_graph=call_g)
    provider = GraphProvider.from_runtime(mock_runtime)

    result = provider.call_graph

    assert result is not None
    assert result.number_of_edges() == 1


def test_provider_import_graph_property() -> None:
    """Import graph property loads and returns graph."""
    import_g = nx.DiGraph()
    import_g.add_edge("m1", "m2")
    mock_runtime = MockGraphRuntime(import_graph=import_g)
    provider = GraphProvider.from_runtime(mock_runtime)

    result = provider.import_graph

    assert result is not None
    assert result.number_of_edges() == 1


def test_provider_symbol_module_graph_property() -> None:
    """Symbol-module graph property loads and returns graph."""
    symbol_g = nx.Graph()
    symbol_g.add_edge("sym1", "mod1")
    mock_runtime = MockGraphRuntime(symbol_module_graph=symbol_g)
    provider = GraphProvider.from_runtime(mock_runtime)

    result = provider.symbol_module_graph

    assert result is not None
    assert result.number_of_edges() == 1


def test_provider_symbol_function_graph_property() -> None:
    """Symbol-function graph property loads and returns graph."""
    symbol_g = nx.Graph()
    symbol_g.add_edge("sym1", "func1")
    mock_runtime = MockGraphRuntime(symbol_function_graph=symbol_g)
    provider = GraphProvider.from_runtime(mock_runtime)

    result = provider.symbol_function_graph

    assert result is not None
    assert result.number_of_edges() == 1


def test_provider_config_module_bipartite_property() -> None:
    """Config-module bipartite property loads and returns graph."""
    config_g = nx.Graph()
    config_g.add_edge("config.key", "mod1")
    mock_runtime = MockGraphRuntime(config_module_bipartite=config_g)
    provider = GraphProvider.from_runtime(mock_runtime)

    result = provider.config_module_bipartite

    assert result is not None
    assert result.number_of_edges() == 1


def test_provider_test_function_bipartite_property() -> None:
    """Test-function bipartite property loads and returns graph."""
    test_g = nx.Graph()
    test_g.add_edge("test_foo", "func1")
    mock_runtime = MockGraphRuntime(test_function_bipartite=test_g)
    provider = GraphProvider.from_runtime(mock_runtime)

    result = provider.test_function_bipartite

    assert result is not None
    assert result.number_of_edges() == 1


def test_provider_cfg_graph_property() -> None:
    """CFG graph property loads and returns graph."""
    cfg_g = nx.DiGraph()
    cfg_g.add_edge("block1", "block2")
    mock_runtime = MockGraphRuntime(cfg_graph=cfg_g)
    provider = GraphProvider.from_runtime(mock_runtime)

    result = provider.cfg_graph

    assert result is not None
    assert result.number_of_edges() == 1


def test_provider_runtime_property() -> None:
    """Runtime property returns None for mock runtime (only returns full GraphRuntime)."""
    mock_runtime = MockGraphRuntime()
    provider = GraphProvider.from_runtime(mock_runtime)

    # runtime property returns None for mocks (only returns full GraphRuntime)
    assert provider.runtime is None
    # runtime_like property returns the mock
    assert provider.runtime_like is mock_runtime


def test_provider_backend_property() -> None:
    """Backend property returns runtime backend."""
    backend_obj = GraphBackendConfig(use_gpu=True)
    mock_runtime = MockGraphRuntime(backend=backend_obj)
    provider = GraphProvider.from_runtime(mock_runtime)

    assert provider.backend is backend_obj


def test_provider_use_gpu_property() -> None:
    """Use GPU property returns runtime setting."""
    mock_runtime = MockGraphRuntime(use_gpu=True)
    provider = GraphProvider.from_runtime(mock_runtime)

    assert provider.use_gpu is True


def test_provider_use_gpu_false_by_default() -> None:
    """Use GPU is False when runtime not built."""
    provider = GraphProvider()

    assert provider.use_gpu is False


def test_provider_backend_none_when_no_runtime() -> None:
    """Backend is None when runtime not built."""
    provider = GraphProvider()

    assert provider.backend is None


def test_provider_without_runtime_raises_on_load() -> None:
    """Provider without runtime or gateway raises ResourceNotLoadedError."""
    provider = GraphProvider()

    with pytest.raises(ResourceNotLoadedError, match="requires either runtime"):
        provider.get()


def test_provider_get_or_none_returns_none_on_error() -> None:
    """Get or None returns None when load fails."""
    provider = GraphProvider()

    result = provider.get_or_none()

    assert result is None


# ============================================================================
# SingleGraphProvider Tests
# ============================================================================


def test_single_graph_loads_call_graph() -> None:
    """SingleGraphProvider loads specific graph type."""
    call_g = nx.DiGraph()
    call_g.add_edge("f1", "f2")
    mock_runtime = MockGraphRuntime(call_graph=call_g)
    parent = GraphProvider.from_runtime(mock_runtime)

    single = SingleGraphProvider(parent, "call_graph")
    result = single.get()

    assert result is not None
    assert result.number_of_edges() == 1


def test_single_graph_raises_if_unavailable() -> None:
    """SingleGraphProvider raises ResourceNotLoadedError if graph not available."""
    mock_runtime = MockGraphRuntime()  # No graphs set
    parent = GraphProvider.from_runtime(mock_runtime)

    single = SingleGraphProvider(parent, "call_graph")

    with pytest.raises(ResourceNotLoadedError, match="not available"):
        single.get()


def test_single_graph_resource_name() -> None:
    """SingleGraphProvider has appropriate resource name."""
    mock_runtime = MockGraphRuntime()
    parent = GraphProvider.from_runtime(mock_runtime)

    single = SingleGraphProvider(parent, "import_graph")

    assert single.resource_name == "Graph:import_graph"


def test_single_graph_caches_result() -> None:
    """SingleGraphProvider caches loaded graph."""
    call_g = nx.DiGraph()
    mock_runtime = MockGraphRuntime(call_graph=call_g)
    parent = GraphProvider.from_runtime(mock_runtime)

    single = SingleGraphProvider(parent, "call_graph")
    result1 = single.get()
    result2 = single.get()

    assert result1 is result2


def test_single_graph_invalidate() -> None:
    """SingleGraphProvider can be invalidated."""
    call_g = nx.DiGraph()
    mock_runtime = MockGraphRuntime(call_graph=call_g)
    parent = GraphProvider.from_runtime(mock_runtime)

    single = SingleGraphProvider(parent, "call_graph")
    _ = single.get()
    assert single.is_loaded

    single.invalidate()
    assert not single.is_loaded
