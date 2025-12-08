"""NetworkX engine behavior mirrors direct nx_views loaders."""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import pytest

from codeintel.graphs.engine import GraphKind, NxGraphEngine
from codeintel.graphs.engine import views as nx_views
from codeintel.graphs.engine.cache import GraphCache
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.context import TestContext
from tests._helpers.factories import make_snapshot
from tests._helpers.seeds import CONFIG_PACK, COVERAGE_PACK, GRAPH_PACK, SYMBOL_PACK


def _node_payload(graph: nx.Graph) -> set[tuple[object, tuple[tuple[str, object], ...]]]:
    return {(node, tuple(sorted(data.items()))) for node, data in graph.nodes(data=True)}


def _edge_payload(graph: nx.Graph) -> set[tuple[object, object, object]]:
    return {(src, dst, data.get("weight", 1)) for src, dst, data in graph.edges(data=True)}


def _assert_graph_match(name: str, expected: nx.Graph, actual: nx.Graph) -> None:
    if _node_payload(expected) != _node_payload(actual):
        pytest.fail(f"{name} nodes differ between engine and nx_views")
    if _edge_payload(expected) != _edge_payload(actual):
        pytest.fail(f"{name} edges differ between engine and nx_views")


def test_engine_matches_nx_views_for_core_graphs(test_ctx: TestContext) -> None:
    """NxGraphEngine should produce the same graphs as direct nx_views loaders."""
    # Apply all required seed packs
    test_ctx.require(GRAPH_PACK, SYMBOL_PACK, CONFIG_PACK, COVERAGE_PACK)

    repo = test_ctx.repo
    commit = test_ctx.commit
    gateway = test_ctx.gateway

    engine = NxGraphEngine(
        gateway=gateway,
        snapshot=make_snapshot(repo=repo, commit=commit),
    )

    comparisons: list[tuple[str, Callable[[], nx.Graph], Callable[[], nx.Graph]]] = [
        (
            "call_graph",
            lambda: nx_views.load_call_graph(gateway, repo, commit),
            engine.call_graph,
        ),
        (
            "import_graph",
            lambda: nx_views.load_import_graph(gateway, repo, commit),
            engine.import_graph,
        ),
        (
            "symbol_module_graph",
            lambda: nx_views.load_symbol_module_graph(gateway, repo, commit),
            engine.symbol_module_graph,
        ),
        (
            "symbol_function_graph",
            lambda: nx_views.load_symbol_function_graph(gateway, repo, commit),
            engine.symbol_function_graph,
        ),
        (
            "config_module_bipartite",
            lambda: nx_views.load_config_module_bipartite(gateway, repo, commit),
            engine.config_module_bipartite,
        ),
        (
            "test_function_bipartite",
            lambda: nx_views.load_test_function_bipartite(gateway, repo, commit),
            engine.test_function_bipartite,
        ),
    ]

    for name, direct_loader, engine_loader in comparisons:
        expected = direct_loader()
        actual = engine_loader()
        _assert_graph_match(name, expected, actual)
        if actual is not engine_loader():
            pytest.fail(f"{name} was not cached on subsequent engine calls")


# ===========================================================================
# GraphCache Tests
# ===========================================================================


def test_cache_seed_with_none_is_noop() -> None:
    """GraphCache.seed with None value does not store anything."""
    cache = GraphCache()

    cache.seed(GraphKind.CALL_GRAPH, None)

    expect_true(cache.has(GraphKind.CALL_GRAPH) is False)


def test_cache_seed_stores_graph() -> None:
    """GraphCache.seed stores the graph for retrieval."""
    cache = GraphCache()
    graph = nx.DiGraph()

    cache.seed(GraphKind.CALL_GRAPH, graph)

    expect_true(cache.has(GraphKind.CALL_GRAPH) is True)


def test_cache_get_returns_cached_graph() -> None:
    """GraphCache.get returns cached graph without calling loader."""
    cache = GraphCache()
    original_graph = nx.DiGraph()
    cache.seed(GraphKind.CALL_GRAPH, original_graph)
    call_count = 0

    def loader() -> nx.DiGraph:
        nonlocal call_count
        call_count += 1
        return nx.DiGraph()

    result = cache.get(GraphKind.CALL_GRAPH, loader)

    expect_true(result is original_graph)
    expect_equal(call_count, 0)


def test_cache_get_calls_loader_when_not_cached() -> None:
    """GraphCache.get calls loader and caches result when not present."""
    cache = GraphCache()
    expected_graph = nx.DiGraph()
    call_count = 0

    def loader() -> nx.DiGraph:
        nonlocal call_count
        call_count += 1
        return expected_graph

    result = cache.get(GraphKind.CALL_GRAPH, loader)

    expected_calls = 1
    expect_true(result is expected_graph)
    expect_equal(call_count, expected_calls)
    expect_true(cache.has(GraphKind.CALL_GRAPH) is True)


def test_cache_clear_removes_all_entries() -> None:
    """GraphCache.clear removes all cached graphs."""
    cache = GraphCache()
    cache.seed(GraphKind.CALL_GRAPH, nx.DiGraph())
    cache.seed(GraphKind.IMPORT_GRAPH, nx.DiGraph())

    cache.clear()

    expect_true(cache.has(GraphKind.CALL_GRAPH) is False)
    expect_true(cache.has(GraphKind.IMPORT_GRAPH) is False)


def test_cache_invalidate_removes_specific_entry() -> None:
    """GraphCache.invalidate removes only the specified graph kind."""
    cache = GraphCache()
    cache.seed(GraphKind.CALL_GRAPH, nx.DiGraph())
    cache.seed(GraphKind.IMPORT_GRAPH, nx.DiGraph())

    cache.invalidate(GraphKind.CALL_GRAPH)

    expect_true(cache.has(GraphKind.CALL_GRAPH) is False)
    expect_true(cache.has(GraphKind.IMPORT_GRAPH) is True)


def test_cache_invalidate_with_missing_key_is_noop() -> None:
    """GraphCache.invalidate with non-existent key does not raise."""
    cache = GraphCache()

    # Should not raise
    cache.invalidate(GraphKind.CALL_GRAPH)
