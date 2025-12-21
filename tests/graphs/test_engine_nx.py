"""NetworkX engine behavior mirrors direct nx_views loaders."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Final

from codeintel.graphs.engine import GraphKind, NxGraphEngine
from codeintel.graphs.engine import views as nx_views
from codeintel.graphs.engine.cache import GraphCache
from tests._helpers.assertions import (
    expect_equal,
    expect_graph_equal,
    expect_is_none,
    expect_true,
)
from tests._helpers.builders import (
    CallGraphNodeRow,
    ConfigValueRow,
    ModuleRow,
    SymbolUseEdgeRow,
    insert_rows,
    insert_symbol_use_edges,
)
from tests._helpers.factories import make_snapshot
from tests._helpers.fakes.networkx_graphs import chain_graph, empty_digraph
from tests._helpers.seeds import CONFIG_PACK, COVERAGE_PACK, GRAPH_PACK, SYMBOL_PACK

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from tests._helpers.context import TestContext

ISOLATED_NODE: Final[int] = 3


def _node_payload(graph: nx.Graph) -> set[tuple[object, tuple[tuple[str, object], ...]]]:
    return {(node, tuple(sorted(data.items()))) for node, data in graph.nodes(data=True)}


def _edge_payload(graph: nx.Graph) -> set[tuple[object, object, object]]:
    return {(src, dst, data.get("weight", 1)) for src, dst, data in graph.edges(data=True)}


def _assert_graph_match(name: str, expected: nx.Graph, actual: nx.Graph) -> None:
    expect_graph_equal(actual, expected, message=f"{name} differs between engine and nx_views")


def test_engine_matches_nx_views_for_core_graphs(test_ctx: TestContext) -> None:
    """NxGraphEngine should produce the same graphs as direct nx_views loaders."""
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
            lambda: nx_views.load_symbol_function_graph(gateway),
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
        expect_true(
            actual is engine_loader(),
            message=f"{name} was not cached on subsequent engine calls",
        )


def test_cache_seed_with_none_is_noop() -> None:
    """GraphCache.seed with None value does not store anything."""
    cache = GraphCache()

    cache.seed(GraphKind.CALL_GRAPH, None)

    expect_true(cache.has(GraphKind.CALL_GRAPH) is False)


def test_cache_seed_stores_graph() -> None:
    """GraphCache.seed stores the graph for retrieval."""
    cache = GraphCache()
    graph = chain_graph(2)

    cache.seed(GraphKind.CALL_GRAPH, graph)

    expect_true(cache.has(GraphKind.CALL_GRAPH) is True)


def test_cache_get_returns_cached_graph() -> None:
    """GraphCache.get returns cached graph without calling loader."""
    cache = GraphCache()
    original_graph = chain_graph(2)
    cache.seed(GraphKind.CALL_GRAPH, original_graph)
    call_count = 0

    def loader() -> nx.DiGraph:
        nonlocal call_count
        call_count += 1
        return empty_digraph()

    result = cache.get(GraphKind.CALL_GRAPH, loader)

    expect_true(result is original_graph)
    expect_equal(call_count, 0)


def test_cache_get_calls_loader_when_not_cached() -> None:
    """GraphCache.get calls loader and caches result when not present."""
    cache = GraphCache()
    expected_graph = chain_graph(2)
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
    cache.seed(GraphKind.CALL_GRAPH, chain_graph(2))
    cache.seed(GraphKind.IMPORT_GRAPH, chain_graph(3))

    cache.clear()

    expect_true(cache.has(GraphKind.CALL_GRAPH) is False)
    expect_true(cache.has(GraphKind.IMPORT_GRAPH) is False)


def test_cache_invalidate_removes_specific_entry() -> None:
    """GraphCache.invalidate removes only the specified graph kind."""
    cache = GraphCache()
    cache.seed(GraphKind.CALL_GRAPH, chain_graph(2))
    cache.seed(GraphKind.IMPORT_GRAPH, chain_graph(3))

    cache.invalidate(GraphKind.CALL_GRAPH)

    expect_true(cache.has(GraphKind.CALL_GRAPH) is False)
    expect_true(cache.has(GraphKind.IMPORT_GRAPH) is True)


def test_cache_invalidate_with_missing_key_is_noop() -> None:
    """GraphCache.invalidate with non-existent key does not raise."""
    cache = GraphCache()

    cache.invalidate(GraphKind.CALL_GRAPH)


def test_numeric_normalizers() -> None:
    """_as_int and _normalize_decimal handle varied inputs."""
    expect_equal(nx_views.as_int(5), 5)
    expect_equal(nx_views.as_int(Decimal("7")), 7)
    expect_equal(nx_views.as_int(b"9"), 9)
    expect_equal(nx_views.as_int(bytearray(b"11")), 11)
    expect_is_none(nx_views.as_int("bad"))
    expect_is_none(nx_views.as_int(b"bad"))
    expect_is_none(nx_views.as_int(None))

    expect_equal(nx_views.normalize_decimal(Decimal("10")), 10)
    expect_equal(nx_views.normalize_decimal(b"12"), 12)
    expect_equal(nx_views.normalize_decimal("14"), 14)
    expect_true(nx_views.normalize_decimal(None) is None)
    expect_true(nx_views.normalize_decimal(object()) is None)


def test_module_attrs_from_row_coerces_values() -> None:
    """_module_attrs_from_row only sets attrs that coerce to int."""
    name, attrs = nx_views.module_attrs_from_row("mod", "1", Decimal("2"), b"3")
    expect_equal(name, "mod")
    expect_equal(attrs["scc_id"], 1)
    expect_equal(attrs["component_size"], 2)
    expect_equal(attrs["layer"], 3)


def test_load_call_graph_weights_and_isolated_nodes(test_ctx: TestContext) -> None:
    """load_call_graph aggregates weights, skips malformed edges, and keeps isolated nodes."""
    repo = test_ctx.repo
    commit = test_ctx.commit
    con = test_ctx.gateway.con
    con.execute("DELETE FROM graph.call_graph_edges")
    con.execute("DELETE FROM graph.call_graph_nodes")

    con.executemany(
        """
        INSERT INTO graph.call_graph_edges
            (repo, commit, caller_goid_h128, callee_goid_h128, callsite_path, callsite_line, callsite_col, language, kind, resolved_via, confidence, evidence_json)
        VALUES (?, ?, ?, ?, 'a.py', 1, 0, 'python', 'call', 'static', 1.0, '{}')
        """,
        [
            (repo, commit, 1, 2),
            (repo, commit, 1, 2),
            (repo, commit, 1, None),
        ],
    )

    insert_rows(
        test_ctx.gateway,
        [
            CallGraphNodeRow(
                goid_h128=3,
                language="python",
                kind="function",
                arity=0,
                is_public=False,
                rel_path="b.py",
            )
        ],
    )

    graph = nx_views.load_call_graph(test_ctx.gateway, repo, commit)

    expect_true(graph.has_edge(1, 2))
    expect_equal(graph[1][2]["weight"], 2)
    expect_true(ISOLATED_NODE in graph.nodes)


def test_load_import_graph_with_missing_import_modules(test_ctx: TestContext) -> None:
    """load_import_graph falls back to module_layer data when import_modules missing."""
    repo = test_ctx.repo
    commit = test_ctx.commit
    con = test_ctx.gateway.con
    con.execute("DELETE FROM graph.import_graph_edges")

    con.executemany(
        """
        INSERT INTO graph.import_graph_edges (repo, commit, src_module, dst_module, src_fan_out, dst_fan_in, cycle_group, module_layer)
        VALUES (?, ?, ?, ?, 0, 0, 0, ?)
        """,
        [
            (repo, commit, "a", "b", 1),
            (repo, commit, "a", "b", 1),
            (repo, commit, "b", "c", 2),
        ],
    )

    con.execute("DROP TABLE IF EXISTS graph.import_modules")

    graph = nx_views.load_import_graph(test_ctx.gateway, repo, commit)

    expect_equal(graph["a"]["b"]["weight"], 2)
    expect_true("a" in graph.nodes)
    expect_equal(graph.nodes["a"]["layer"], 1)


def test_load_test_function_bipartite_weights(test_ctx: TestContext) -> None:
    """load_test_function_bipartite accumulates weights and skips null IDs."""
    repo = test_ctx.repo
    commit = test_ctx.commit
    con = test_ctx.gateway.con
    con.execute("DELETE FROM analytics.test_coverage_edges")
    con.executemany(
        """
        INSERT INTO analytics.test_coverage_edges (repo, commit, test_id, function_goid_h128, coverage_ratio)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (repo, commit, "t1", 1, 0.5),
            (repo, commit, "t1", 1, 0.25),
            (repo, commit, None, 2, 0.5),
        ],
    )

    graph = nx_views.load_test_function_bipartite(test_ctx.gateway, repo, commit)

    test_node = ("t", "t1")
    func_node = ("f", 1)
    expect_true(graph.has_node(test_node))
    expect_true(graph.has_node(func_node))
    expect_equal(graph[test_node][func_node]["weight"], 0.75)


def test_parse_reference_modules_and_config_bipartite(test_ctx: TestContext) -> None:
    """_parse_reference_modules filters and load_config_module_bipartite keeps raw when filter drops all."""
    expect_equal(nx_views.parse_reference_modules(["a", "b"], {"a"}), ["a"])
    expect_equal(nx_views.parse_reference_modules('["a","b"]', set()), ["a", "b"])
    expect_equal(nx_views.parse_reference_modules("bad json", {"a"}), [])

    repo = test_ctx.repo
    commit = test_ctx.commit
    con = test_ctx.gateway.con
    con.execute("DELETE FROM analytics.config_values")
    con.execute("DELETE FROM core.modules")

    insert_rows(
        test_ctx.gateway,
        [
            ModuleRow(module="allowed", path="pkg/allowed.py", repo=repo, commit=commit),
        ],
    )
    insert_rows(
        test_ctx.gateway,
        [
            ConfigValueRow(
                repo=repo,
                commit=commit,
                config_path="cfg1",
                format="json",
                key="k1",
                reference_paths=[],
                reference_modules=["missing.mod"],
                reference_count=1,
            ),
            ConfigValueRow(
                repo=repo,
                commit=commit,
                config_path="cfg2",
                format="json",
                key="k2",
                reference_paths=[],
                reference_modules=["allowed"],
                reference_count=1,
            ),
        ],
    )

    graph = nx_views.load_config_module_bipartite(test_ctx.gateway, repo, commit)

    expect_true(("c", "k1") in graph.nodes)
    expect_true(("m", "missing.mod") in graph.nodes)
    expect_true(graph.has_edge(("c", "k2"), ("m", "allowed")))


def test_load_symbol_module_graph_weights(test_ctx: TestContext) -> None:
    """load_symbol_module_graph skips self-edges and increments weights."""
    repo = test_ctx.repo
    commit = test_ctx.commit
    con = test_ctx.gateway.con
    con.execute("DELETE FROM graph.symbol_use_edges")
    con.execute("DELETE FROM core.modules")

    insert_rows(
        test_ctx.gateway,
        [
            ModuleRow(module="m_a", path="a.py", repo=repo, commit=commit),
            ModuleRow(module="m_b", path="b.py", repo=repo, commit=commit),
        ],
    )
    insert_rows(
        test_ctx.gateway,
        [
            SymbolUseEdgeRow(
                symbol="s", def_path="a.py", use_path="b.py", same_file=False, same_module=False
            ),
            SymbolUseEdgeRow(
                symbol="t", def_path="a.py", use_path="b.py", same_file=False, same_module=False
            ),
            SymbolUseEdgeRow(
                symbol="self", def_path="a.py", use_path="a.py", same_file=True, same_module=True
            ),
        ],
    )

    graph = nx_views.load_symbol_module_graph(test_ctx.gateway, repo, commit)

    expect_true(graph.has_edge("m_b", "m_a"))
    expect_equal(graph["m_b"]["m_a"]["weight"], 2)


def test_load_symbol_function_graph_handles_duckdb_error_and_normalization(
    test_ctx: TestContext,
) -> None:
    """load_symbol_function_graph returns empty on DuckDBError and normalizes decimals."""
    con = test_ctx.gateway.con
    con.execute("DELETE FROM graph.symbol_use_edges")

    insert_symbol_use_edges(
        test_ctx.gateway,
        [
            (
                "s1",
                "a.py",
                "b.py",
                False,
                False,
                Decimal("10"),
                20,
            ),
        ],
    )

    graph = nx_views.load_symbol_function_graph(test_ctx.gateway)
    expect_true(graph.has_edge(10, 20))

    con.execute("DROP TABLE IF EXISTS graph.symbol_use_edges")
    empty_graph = nx_views.load_symbol_function_graph(test_ctx.gateway)
    expect_equal(empty_graph.number_of_nodes(), 0)
