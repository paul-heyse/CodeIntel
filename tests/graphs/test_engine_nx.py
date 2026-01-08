"""Graph engine behavior mirrors direct graph view loaders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from codeintel.build.graphs.engine import GraphKind, RxGraphEngine
from codeintel.build.graphs.engine import views as graph_views
from codeintel.build.graphs.engine.cache import GraphCache
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.core.datasets.arrow_store import write_dataset
from tests._helpers.assertions import (
    assert_target_ok,
    expect_equal,
    expect_graph_equal,
    expect_is_none,
    expect_true,
)
from tests._helpers.columnar_streams import table_for_rows
from tests._helpers.fixtures.graphs import chain_graph, empty_digraph
from tests._helpers.fixtures.rows import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    ConfigValueRow,
    ImportGraphEdgeRow,
    ModuleRow,
    SymbolUseEdgeRow,
)
from tests._helpers.harnesses.graph_harness import GraphTargetHarness

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests._helpers.builders.row_protocol import InsertableRow
    from tests._helpers.context import TestContext

ISOLATED_NODE: Final[int] = 3


def _assert_graph_match(name: str, expected: RxGraphStore, actual: RxGraphStore) -> None:
    expect_graph_equal(actual, expected, message=f"{name} differs between engine and views")


def _edge_weight(store: RxGraphStore, src: object, dst: object) -> float | None:
    src_idx = store.get_index(src)
    dst_idx = store.get_index(dst)
    if src_idx is None or dst_idx is None:
        return None
    if not store.graph.has_edge(src_idx, dst_idx):
        return None
    payload = store.graph.get_edge_data(src_idx, dst_idx)
    return edge_weight_from_payload(payload)


def _has_node(store: RxGraphStore, node_id: object) -> bool:
    return store.get_index(node_id) is not None


def _write_dataset_rows(
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    rows: Sequence[InsertableRow],
) -> None:
    if not rows:
        return
    dataset_root.mkdir(parents=True, exist_ok=True)
    columns = type(rows[0]).__columns__
    table = table_for_rows(
        table_key,
        [_row_mapping(columns, row.to_tuple()) for row in rows],
        columns=columns,
    )
    write_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        data=table,
    )


def _write_dataset_mappings(
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        return
    dataset_root.mkdir(parents=True, exist_ok=True)
    table = table_for_rows(table_key, rows)
    write_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        data=table,
    )


def _row_mapping(
    columns: Sequence[str],
    values: Sequence[object],
) -> dict[str, object]:
    return dict(zip(columns, values, strict=True))


def test_engine_matches_views_for_core_graphs(
    graph_target_harness: GraphTargetHarness,
) -> None:
    """RxGraphEngine should produce the same graphs as direct views loaders.

    Raises
    ------
    ValueError
        If graph target execution fails for reasons other than schema availability.
    """
    try:
        records = graph_target_harness.run_targets()
    except ValueError as exc:
        if "Missing TableSchema definitions for DAG outputs" in str(exc):
            pytest.xfail("Schema registry incomplete for graph targets in this runtime.")
        raise
    assert_target_ok(records["call_graph"])
    assert_target_ok(records["import_graph"])

    snapshot = graph_target_harness.harness.ctx.snapshot
    dataset_root = graph_target_harness.harness.ctx.build_paths.dataset_root_dir
    engine = RxGraphEngine(
        dataset_root_dir=dataset_root,
        snapshot=snapshot,
    )

    comparisons: list[tuple[str, Callable[[], RxGraphStore], Callable[[], RxGraphStore]]] = [
        (
            "call_graph",
            lambda: graph_views.load_call_graph(dataset_root, snapshot.repo, snapshot.commit),
            engine.call_graph,
        ),
        (
            "import_graph",
            lambda: graph_views.load_import_graph(dataset_root, snapshot.repo, snapshot.commit),
            engine.import_graph,
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


def test_engine_matches_harness_graph_targets(graph_target_harness: GraphTargetHarness) -> None:
    """RxGraphEngine should match views after harness graph target runs.

    Raises
    ------
    ValueError
        If graph target execution fails for reasons other than schema availability.
    """
    try:
        records = graph_target_harness.run_targets()
    except ValueError as exc:
        if "Missing TableSchema definitions for DAG outputs" in str(exc):
            pytest.xfail("Schema registry incomplete for graph targets in this runtime.")
        raise
    assert_target_ok(records["call_graph"])
    assert_target_ok(records["import_graph"])

    snapshot = graph_target_harness.harness.ctx.snapshot
    dataset_root = graph_target_harness.harness.ctx.build_paths.dataset_root_dir
    engine = RxGraphEngine(
        dataset_root_dir=dataset_root,
        snapshot=snapshot,
    )
    expected_call = graph_views.load_call_graph(dataset_root, snapshot.repo, snapshot.commit)
    expected_import = graph_views.load_import_graph(dataset_root, snapshot.repo, snapshot.commit)
    _assert_graph_match("call_graph", expected_call, engine.call_graph())
    _assert_graph_match("import_graph", expected_import, engine.import_graph())


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

    def loader() -> RxGraphStore:
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

    def loader() -> RxGraphStore:
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
    expect_equal(graph_views.as_int(5), 5)
    expect_equal(graph_views.as_int(Decimal("7")), 7)
    expect_equal(graph_views.as_int(b"9"), 9)
    expect_equal(graph_views.as_int(bytearray(b"11")), 11)
    expect_is_none(graph_views.as_int("bad"))
    expect_is_none(graph_views.as_int(b"bad"))
    expect_is_none(graph_views.as_int(None))

    expect_equal(graph_views.normalize_decimal(Decimal("10")), 10)
    expect_equal(graph_views.normalize_decimal(b"12"), 12)
    expect_equal(graph_views.normalize_decimal("14"), 14)
    expect_true(graph_views.normalize_decimal(None) is None)
    expect_true(graph_views.normalize_decimal(object()) is None)


def test_module_attrs_from_row_coerces_values() -> None:
    """_module_attrs_from_row only sets attrs that coerce to int."""
    name, attrs = graph_views.module_attrs_from_row("mod", "1", Decimal("2"), b"3")
    expect_equal(name, "mod")
    expect_equal(attrs["scc_id"], 1)
    expect_equal(attrs["component_size"], 2)
    expect_equal(attrs["layer"], 3)


def test_load_call_graph_weights_and_isolated_nodes(test_ctx: TestContext) -> None:
    """load_call_graph aggregates weights, skips malformed edges, and keeps isolated nodes."""
    repo = test_ctx.repo
    commit = test_ctx.commit
    dataset_root = test_ctx.build_paths.dataset_root_dir
    edges = [
        CallGraphEdgeRow(
            repo=repo,
            commit=commit,
            caller_goid_h128=1,
            callee_goid_h128=2,
            callsite_path="a.py",
            callsite_line=1,
            callsite_col=0,
            language="python",
            kind="call",
            resolved_via="static",
            confidence=1.0,
        ),
        CallGraphEdgeRow(
            repo=repo,
            commit=commit,
            caller_goid_h128=1,
            callee_goid_h128=2,
            callsite_path="a.py",
            callsite_line=1,
            callsite_col=0,
            language="python",
            kind="call",
            resolved_via="static",
            confidence=1.0,
        ),
        CallGraphEdgeRow(
            repo=repo,
            commit=commit,
            caller_goid_h128=1,
            callee_goid_h128=None,
            callsite_path="a.py",
            callsite_line=1,
            callsite_col=0,
            language="python",
            kind="call",
            resolved_via="static",
            confidence=1.0,
        ),
    ]
    nodes = [
        CallGraphNodeRow(
            goid_h128=3,
            language="python",
            kind="function",
            arity=0,
            is_public=False,
            rel_path="b.py",
        )
    ]
    _write_dataset_rows(dataset_root, "graph.call_graph_edges", commit, edges)
    _write_dataset_rows(dataset_root, "graph.call_graph_nodes", commit, nodes)

    graph = graph_views.load_call_graph(dataset_root, repo, commit)

    expect_true(_edge_weight(graph, 1, 2) is not None)
    expect_equal(_edge_weight(graph, 1, 2), 2.0)
    expect_true(_has_node(graph, ISOLATED_NODE))


def test_load_import_graph_with_missing_import_modules(test_ctx: TestContext) -> None:
    """load_import_graph falls back to module_layer data when import_modules missing."""
    repo = test_ctx.repo
    commit = test_ctx.commit
    dataset_root = test_ctx.build_paths.dataset_root_dir
    columns = (*ImportGraphEdgeRow.__columns__, "module_layer")
    rows = [
        _row_mapping(columns, (repo, commit, "a", "b", 0, 0, 0, 1)),
        _row_mapping(columns, (repo, commit, "a", "b", 0, 0, 0, 1)),
        _row_mapping(columns, (repo, commit, "b", "c", 0, 0, 0, 2)),
    ]
    _write_dataset_mappings(
        dataset_root,
        "graph.import_graph_edges",
        commit,
        rows,
    )

    graph = graph_views.load_import_graph(dataset_root, repo, commit)

    expect_equal(_edge_weight(graph, "a", "b"), 2.0)
    expect_true(_has_node(graph, "a"))
    expect_equal(graph.get_node_attrs("a")["layer"], 1)


def test_parse_reference_modules_and_config_bipartite(test_ctx: TestContext) -> None:
    """_parse_reference_modules filters and load_config_module_bipartite keeps raw when filter drops all."""
    expect_equal(graph_views.parse_reference_modules(["a", "b"], {"a"}), ["a"])
    expect_equal(graph_views.parse_reference_modules('["a","b"]', set()), ["a", "b"])
    expect_equal(graph_views.parse_reference_modules("bad json", {"a"}), [])

    repo = test_ctx.repo
    commit = test_ctx.commit
    dataset_root = test_ctx.build_paths.dataset_root_dir
    modules = [ModuleRow(module="allowed", path="pkg/allowed.py", repo=repo, commit=commit)]
    configs = [
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
    ]
    _write_dataset_rows(dataset_root, "core.modules", commit, modules)
    _write_dataset_rows(dataset_root, "analytics.config_values", commit, configs)

    graph = graph_views.load_config_module_bipartite(dataset_root, repo, commit)

    expect_true(_has_node(graph, ("c", "k1")))
    expect_true(_has_node(graph, ("m", "missing.mod")) is False)
    expect_true(_has_node(graph, ("m", "allowed")) is False)
    expect_true(_edge_weight(graph, ("c", "k2"), ("m", "allowed")) is None)


def test_load_symbol_module_graph_weights(test_ctx: TestContext) -> None:
    """load_symbol_module_graph skips self-edges and increments weights."""
    repo = test_ctx.repo
    commit = test_ctx.commit
    dataset_root = test_ctx.build_paths.dataset_root_dir
    modules = [
        ModuleRow(module="m_a", path="a.py", repo=repo, commit=commit),
        ModuleRow(module="m_b", path="b.py", repo=repo, commit=commit),
    ]
    edges = [
        SymbolUseEdgeRow(
            repo=repo,
            commit=commit,
            symbol="s",
            def_path="a.py",
            use_path="b.py",
            same_file=False,
            same_module=False,
        ),
        SymbolUseEdgeRow(
            repo=repo,
            commit=commit,
            symbol="t",
            def_path="a.py",
            use_path="b.py",
            same_file=False,
            same_module=False,
        ),
        SymbolUseEdgeRow(
            repo=repo,
            commit=commit,
            symbol="self",
            def_path="a.py",
            use_path="a.py",
            same_file=True,
            same_module=True,
        ),
    ]
    _write_dataset_rows(dataset_root, "core.modules", commit, modules)
    _write_dataset_rows(dataset_root, "graph.symbol_use_edges", commit, edges)

    graph = graph_views.load_symbol_module_graph(dataset_root, repo, commit)

    expect_true(_edge_weight(graph, "m_b", "m_a") is not None)
    expect_equal(_edge_weight(graph, "m_b", "m_a"), 2.0)


def test_load_symbol_function_graph_handles_duckdb_error_and_normalization(
    test_ctx: TestContext,
) -> None:
    """load_symbol_function_graph returns empty when dataset missing and normalizes decimals."""
    commit = test_ctx.commit
    dataset_root = test_ctx.build_paths.dataset_root_dir
    columns = SymbolUseEdgeRow.__columns__
    rows = [
        _row_mapping(
            columns,
            (
                test_ctx.repo,
                commit,
                "s1",
                "a.py",
                "b.py",
                False,
                False,
                Decimal("10"),
                20,
            ),
        ),
    ]
    _write_dataset_mappings(
        dataset_root,
        "graph.symbol_use_edges",
        commit,
        rows,
    )

    graph = graph_views.load_symbol_function_graph(dataset_root, commit)
    expect_true(_edge_weight(graph, 10, 20) is not None)

    missing_root = dataset_root / "missing"
    empty_graph = graph_views.load_symbol_function_graph(missing_root, commit)
    expect_equal(empty_graph.graph.num_nodes(), 0)
