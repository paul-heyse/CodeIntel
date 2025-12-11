"""Graph runtime cache, pooling, and graph helper tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import networkx as nx
import pytest
from networkx.readwrite import json_graph

from codeintel.analytics.compute.graphs import (
    bipartite_degrees,
    bounded_simple_path_count,
    centrality_directed,
    centrality_undirected,
    component_metadata,
    global_graph_stats,
    neighbor_stats,
    projection_metrics,
    structural_metrics,
)
from codeintel.analytics.graphs.graph_metrics import GraphMetricsDeps, compute_graph_metrics
from codeintel.analytics.runtime import (
    GraphRuntimeOptions,
    GraphRuntimePool,
    build_graph_runtime,
)
from codeintel.analytics.runtime.context import GraphContext
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig
from codeintel.graphs.core.registry import get_graph_registry
from codeintel.graphs.engine import GraphKind
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_true,
)
from tests._helpers.fakes.graph_plugins import GraphPluginBuilder, plugin_registrar

if TYPE_CHECKING:
    from codeintel.analytics.compute.graphs import (
        BipartiteDegrees,
    )
    from tests._helpers.graph_runtime_harness import GraphRuntimeHarness


def test_disk_cache_round_trip(graph_runtime_ctx: GraphRuntimeHarness) -> None:
    """Graphs should be loaded from disk cache when metadata matches."""
    engine = graph_runtime_ctx.build_engine()
    runtime = graph_runtime_ctx.build_runtime(engine=engine)

    graph1 = runtime.ensure_call_graph()
    expect_equal(engine.method_counts.get("load_call_graph", 0), 1)
    expect_equal(
        graph1.number_of_nodes(),
        graph_runtime_ctx.fixtures.call_graph.number_of_nodes(),
    )

    engine2 = graph_runtime_ctx.build_engine()
    runtime2 = graph_runtime_ctx.build_runtime(engine=engine2)
    graph2 = runtime2.ensure_call_graph()
    expect_equal(engine2.method_counts.get("load_call_graph", 0), 0)
    expect_equal(
        graph2.number_of_edges(),
        graph_runtime_ctx.fixtures.call_graph.number_of_edges(),
    )


def test_disk_cache_mismatch_falls_back_to_loader(graph_runtime_ctx: GraphRuntimeHarness) -> None:
    """Cache metadata mismatch should trigger loader path."""
    engine = graph_runtime_ctx.build_engine()
    runtime = graph_runtime_ctx.build_runtime(engine=engine)

    safe_repo = graph_runtime_ctx.snapshot.repo.replace("/", "__")
    use_gpu_str = "true" if runtime.backend.use_gpu else "false"
    base = graph_runtime_ctx.cache_dir / (
        f"{safe_repo}__{graph_runtime_ctx.snapshot.commit}"
        f"__{runtime.backend.backend}__{use_gpu_str}__{str(GraphKind.CALL_GRAPH).lower()}"
    )
    graph_path = base.with_suffix(".json")
    meta_path = base.with_suffix(".meta")
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_payload = json_graph.node_link_data(nx.DiGraph())
    graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")
    meta_path.write_text(
        "\n".join(["other", "c", runtime.backend.backend, "false"]), encoding="utf-8"
    )

    runtime.ensure_call_graph()
    expect_equal(engine.method_counts.get("load_call_graph", 0), 1)


def test_compute_graph_metrics_reuses_runtime_engine(
    graph_runtime_ctx: GraphRuntimeHarness,
) -> None:
    """Provided GraphRuntime should be reused without rebuilding the engine."""
    runtime = build_graph_runtime(
        graph_runtime_ctx.gateway,
        GraphRuntimeOptions(snapshot=graph_runtime_ctx.snapshot),
    )
    compute_graph_metrics(
        graph_runtime_ctx.gateway,
        GraphMetricsStepConfig(snapshot=graph_runtime_ctx.snapshot),
        deps=GraphMetricsDeps(runtime=runtime),
    )
    expect_true(runtime.engine is not None)


def test_pool_reuses_runtime_within_ttl(graph_runtime_ctx: GraphRuntimeHarness) -> None:
    """Pool should return the same runtime when within TTL."""
    apply_all_schemas(graph_runtime_ctx.gateway.con)
    options = GraphRuntimeOptions(snapshot=graph_runtime_ctx.snapshot)
    current = [0.0]

    def _time() -> float:
        return current[0]

    pool = GraphRuntimePool(ttl_seconds=10.0, time_func=_time)
    first = pool.get(graph_runtime_ctx.gateway, options)
    current[0] += 1.0
    second = pool.get(graph_runtime_ctx.gateway, options)
    expect_true(first is second)


def test_pool_expires_runtime_after_ttl(graph_runtime_ctx: GraphRuntimeHarness) -> None:
    """Pool should rebuild runtime after TTL expires."""
    apply_all_schemas(graph_runtime_ctx.gateway.con)
    options = GraphRuntimeOptions(snapshot=graph_runtime_ctx.snapshot)
    current = [0.0]

    def _time() -> float:
        return current[0]

    pool = GraphRuntimePool(ttl_seconds=0.5, time_func=_time)
    first = pool.get(graph_runtime_ctx.gateway, options)
    current[0] += 1.0
    second = pool.get(graph_runtime_ctx.gateway, options)
    expect_true(first is not second)


def test_pool_lru_eviction(graph_runtime_ctx: GraphRuntimeHarness) -> None:
    """LRU eviction should drop least-recently-used runtime when capacity exceeded."""
    apply_all_schemas(graph_runtime_ctx.gateway.con)
    snapshot2 = SnapshotRef(
        repo=graph_runtime_ctx.snapshot.repo,
        commit="other",
        repo_root=graph_runtime_ctx.snapshot.repo_root,
    )
    opts1 = GraphRuntimeOptions(snapshot=graph_runtime_ctx.snapshot)
    opts2 = GraphRuntimeOptions(snapshot=snapshot2)
    current = [0.0]

    def _time() -> float:
        return current[0]

    pool = GraphRuntimePool(max_size=1, time_func=_time)
    first = pool.get(graph_runtime_ctx.gateway, opts1)
    current[0] += 1.0
    pool.get(graph_runtime_ctx.gateway, opts2)
    current[0] += 1.0
    again = pool.get(graph_runtime_ctx.gateway, opts1)
    expect_true(again is not first)


def test_plugin_registrar_registers_plugins() -> None:
    """plugin_registrar should register and then unregister plugins."""
    plugin = GraphPluginBuilder(name="unit.plugin").with_provides("capability").build()
    registry = get_graph_registry()
    expect_false(registry.contains(plugin.metadata.name))
    with plugin_registrar([plugin]):
        expect_true(registry.contains(plugin.metadata.name))
        names = [p.metadata.name for p in registry.list_all()]
        expect_in(plugin.metadata.name, names)
    expect_false(registry.contains(plugin.metadata.name))


def test_centrality_directed_weighted_and_seeded() -> None:
    """Directed centrality respects weights and seeded randomness."""
    graph = nx.DiGraph()
    graph.add_edge(1, 2, weight=2.0)
    graph.add_edge(2, 3, weight=1.0)
    ctx = GraphContext(repo="r", commit="c", now=datetime.now(tz=UTC), seed=42)
    metrics = centrality_directed(graph, ctx)
    if metrics.betweenness[2] <= metrics.betweenness.get(1, 0.0):
        pytest.fail("Expected node 2 betweenness to exceed node 1")
    if metrics.pagerank[1] == metrics.pagerank[2]:
        pytest.fail("Expected pagerank variation across nodes")


def test_centrality_undirected_empty_returns_zeroes() -> None:
    """Undirected centrality returns empty metrics for empty graph."""
    graph = nx.Graph()
    ctx = GraphContext(repo="r", commit="c", now=datetime.now(tz=UTC))
    metrics = centrality_undirected(graph, ctx)
    if metrics.betweenness or metrics.pagerank:
        pytest.fail("Expected empty centrality metrics for empty graph")


def test_neighbor_stats_with_weights() -> None:
    """Neighbor stats aggregate weighted degrees."""
    graph = nx.DiGraph()
    weighted_edge_value = 3
    graph.add_edge("a", "b", weight=weighted_edge_value)
    stats = neighbor_stats(graph, weight="weight")
    if stats.out_counts.get("a") != weighted_edge_value:
        pytest.fail(
            f"Expected weighted out count of {weighted_edge_value}, got {stats.out_counts.get('a')}"
        )
    if "b" not in stats.in_neighbors:
        pytest.fail("Expected inbound neighbor entry for 'b'")


def test_component_metadata_cycles_and_layers() -> None:
    """Component metadata marks cycles and assigns layers."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 1), (2, 3)])
    meta = component_metadata(graph)
    if not meta.in_cycle.get(1):
        pytest.fail("Node 1 should be marked as part of a cycle")
    if meta.layer.get(3, 0) < meta.layer.get(1, 0):
        pytest.fail("Downstream node should not have lower layer than cycle node")


def test_projection_metrics_basic() -> None:
    """Projection metrics compute weighted degree for projected nodes."""
    graph = nx.Graph()
    graph.add_edge(("x", 1), ("y", 1), weight=1.0)
    graph.add_edge(("x", 2), ("y", 1), weight=2.0)
    ctx = GraphContext(repo="r", commit="c", now=datetime.now(tz=UTC), seed=0)
    metrics = projection_metrics(graph, {("x", 1), ("x", 2)}, ctx)
    weighted_degree = metrics.weighted_degree.get(("x", 1), 0.0)
    if weighted_degree <= 0.0:
        pytest.fail(f"Expected projected weighted degree to be positive, got {weighted_degree}")
    closeness = metrics.closeness.get(("x", 1), 0.0)
    if closeness <= 0.0:
        pytest.fail(f"Expected closeness to be computed, got {closeness}")
    if metrics.community_id.get(("x", 1)) is None:
        pytest.fail("Expected community assignment for projected node")


def test_structural_metrics_triangle_and_core() -> None:
    """Structural metrics include triangle counts and k-core values."""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 1)])
    metrics = structural_metrics(graph)
    expected_triangles = 1
    if metrics.triangles.get(1) != expected_triangles:
        pytest.fail(f"Unexpected triangle count: {metrics.triangles.get(1)}")
    if metrics.core_number.get(1, 0) < 1:
        pytest.fail("Expected node 1 to be part of core >= 1")


def test_global_graph_stats_triangle() -> None:
    """Global stats summarize undirected graphs."""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 1)])
    expected_nodes = 3
    expected_edges = 3
    expected_components = 1
    expected_distance = 1.0
    stats = global_graph_stats(graph)
    if stats.node_count != expected_nodes or stats.edge_count != expected_edges:
        pytest.fail("Unexpected node or edge counts in global stats")
    if stats.weak_component_count != expected_components or stats.scc_count != expected_components:
        pytest.fail("Expected single component for triangle graph")
    if stats.component_layers is not None:
        pytest.fail("Undirected graphs should not report component layers")
    if stats.diameter_estimate != pytest.approx(expected_distance):
        pytest.fail("Unexpected diameter or average shortest path for triangle")
    if stats.avg_shortest_path_estimate != pytest.approx(expected_distance):
        pytest.fail("Unexpected average shortest path for triangle graph")
    if stats.avg_clustering != pytest.approx(1.0):
        pytest.fail("Expected full clustering for triangle graph")


def test_bipartite_degrees_weight_and_centrality() -> None:
    """Bipartite degrees include weighted counts and partition-specific centralities."""
    graph = nx.Graph()
    test_node = ("t", 1)
    func_node = ("f", 1)
    graph.add_edge(test_node, func_node, weight=2.0)
    metrics: BipartiteDegrees = bipartite_degrees(graph, {test_node}, {func_node}, weight="weight")
    if metrics.weighted_degree.get(test_node) != pytest.approx(2.0):
        pytest.fail("Weighted degree did not reflect edge weight")
    if metrics.primary_degree_centrality.get(test_node, 0.0) <= 0.0:
        pytest.fail("Expected non-zero degree centrality for populated bipartite graph")


def test_bounded_simple_path_count_caps_results() -> None:
    """Simple path counting respects the maximum path limit."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    cap = 1
    expected_paths = 2
    capped = bounded_simple_path_count(
        graph,
        {1},
        {4},
        max_paths=cap,
        cutoff=5,
    )
    if capped != cap:
        pytest.fail(f"Expected capped count of {cap}, got {capped}")
    uncapped = bounded_simple_path_count(
        graph,
        {1},
        {4},
        max_paths=10,
        cutoff=5,
    )
    if uncapped != expected_paths:
        pytest.fail(f"Expected to discover both simple paths, got {uncapped}")
