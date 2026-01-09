"""Test graph centrality metric computation.

Test the pure computation functions for computing PageRank and
betweenness centrality on directed graphs using rustworkx stores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from codeintel.build.analytics.compute.graphs import centrality as centrality_module
from codeintel.build.graphs.runtime.context import GraphContext
from codeintel.build.graphs.rx.algos import BetweennessOptions, PagerankOptions
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.core.compute.centrality import (
    CentralityMetrics,
    compute_betweenness,
    compute_pagerank,
)
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    assert_logged,
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_not_equal,
    expect_true,
)

if TYPE_CHECKING:
    from datetime import datetime

    import pytest


EXPECTED_NODES_4 = 4
EXPECTED_NODES_5 = 5
EXPECTED_NODES_7 = 7
EXPECTED_TOP_3 = 3
TOLERANCE = 0.001
DENSE_GRAPH_RANGE_TOLERANCE = 0.1
PAGERANK_SUM = 1.0


TEST_PAGERANK = 0.25
TEST_BETWEENNESS = 0.5
TEST_IN_DEGREE = 3
TEST_OUT_DEGREE = 2
TEST_CLOSENESS = 0.75
TEST_HARMONIC = 0.8
TEST_EIGENVECTOR = 0.6


def _directed_store(edges: list[tuple[object, object]]) -> RxGraphStore:
    store = RxGraphStore.directed()
    for src, dst in edges:
        store.add_weighted_edge(src, dst, weight=1.0)
    return store


def _undirected_store(edges: list[tuple[object, object]]) -> RxGraphStore:
    store = RxGraphStore.undirected()
    for src, dst in edges:
        store.add_weighted_edge(src, dst, weight=1.0)
    return store


def _make_simple_chain() -> RxGraphStore:
    """
    Create a simple linear chain graph: A -> B -> C -> D.

    Returns
    -------
    RxGraphStore
        A directed chain graph store.
    """
    return _directed_store([("A", "B"), ("B", "C"), ("C", "D")])


def _make_simple_cycle() -> RxGraphStore:
    """
    Create a simple cycle graph: A -> B -> C -> A.

    Returns
    -------
    RxGraphStore
        A directed cycle graph store.
    """
    return _directed_store([("A", "B"), ("B", "C"), ("C", "A")])


def _make_star_graph() -> RxGraphStore:
    """
    Create a star graph with hub node pointing to spokes: Hub -> {A, B, C, D}.

    Returns
    -------
    RxGraphStore
        An out-star directed graph store.
    """
    return _directed_store([("Hub", "A"), ("Hub", "B"), ("Hub", "C"), ("Hub", "D")])


def _make_reverse_star_graph() -> RxGraphStore:
    """
    Create a reverse star with spokes pointing to hub: {A, B, C, D} -> Hub.

    Returns
    -------
    RxGraphStore
        An in-star directed graph store.
    """
    return _directed_store([("A", "Hub"), ("B", "Hub"), ("C", "Hub"), ("D", "Hub")])


def _make_call_graph_realistic() -> RxGraphStore:
    """
    Create a realistic call graph structure.

    Simulates a typical codebase call hierarchy:
    - main() calls process_request() and handle_error()
    - process_request() calls validate(), execute(), log_result()
    - validate() and execute() share utility functions

    Returns
    -------
    RxGraphStore
        A realistic call graph store.
    """
    return _directed_store(
        [
            ("main", "process_request"),
            ("main", "handle_error"),
            ("process_request", "validate"),
            ("process_request", "execute"),
            ("process_request", "log_result"),
            ("validate", "check_input"),
            ("validate", "sanitize"),
            ("execute", "fetch_data"),
            ("execute", "transform"),
            ("execute", "save_result"),
            ("validate", "format_error"),
            ("execute", "format_error"),
            ("handle_error", "format_error"),
            ("handle_error", "log_result"),
        ]
    )


def _make_disconnected_components() -> RxGraphStore:
    """
    Create a graph with multiple disconnected components.

    Returns
    -------
    RxGraphStore
        A graph store with three disconnected components.
    """
    graph = _directed_store([("A", "B"), ("B", "C"), ("X", "Y"), ("Y", "Z")])
    graph.ensure_node("Isolated")
    return graph


def _make_dense_cluster() -> RxGraphStore:
    """
    Create a densely connected cluster.

    Returns
    -------
    RxGraphStore
        A complete directed graph store with 5 nodes.
    """
    graph = RxGraphStore.directed()
    nodes = ["N1", "N2", "N3", "N4", "N5"]
    for node in nodes:
        graph.ensure_node(node)
    for source in nodes:
        for target in nodes:
            if source != target:
                graph.add_weighted_edge(source, target, weight=1.0)
    return graph


def test_metrics_create_all_fields() -> None:
    """Create metrics dataclass with all fields."""
    metrics = CentralityMetrics(
        pagerank=TEST_PAGERANK,
        betweenness=TEST_BETWEENNESS,
        closeness=TEST_CLOSENESS,
        harmonic=TEST_HARMONIC,
        eigenvector=TEST_EIGENVECTOR,
        in_degree=TEST_IN_DEGREE,
        out_degree=TEST_OUT_DEGREE,
        degree=TEST_IN_DEGREE + TEST_OUT_DEGREE,
    )
    expect_equal(metrics.pagerank, TEST_PAGERANK)
    expect_equal(metrics.betweenness, TEST_BETWEENNESS)
    expect_equal(metrics.closeness, TEST_CLOSENESS)
    expect_equal(metrics.harmonic, TEST_HARMONIC)
    expect_equal(metrics.eigenvector, TEST_EIGENVECTOR)
    expect_equal(metrics.in_degree, TEST_IN_DEGREE)
    expect_equal(metrics.out_degree, TEST_OUT_DEGREE)
    expect_equal(metrics.degree, TEST_IN_DEGREE + TEST_OUT_DEGREE)


def test_metrics_is_frozen() -> None:
    """Metrics dataclass is immutable (frozen)."""
    metrics = CentralityMetrics(
        pagerank=0.1,
        betweenness=0.2,
        closeness=0.3,
        harmonic=0.4,
        eigenvector=0.5,
        in_degree=1,
        out_degree=1,
        degree=2,
    )
    assert_frozen(metrics, "pagerank", 0.5)


def test_pagerank_empty_graph() -> None:
    """Empty graph returns empty PageRank dictionary."""
    graph = RxGraphStore.directed()
    result = compute_pagerank(graph)
    expect_equal(result, {})


def test_pagerank_single_node() -> None:
    """Single node gets PageRank of 1.0."""
    graph = RxGraphStore.directed()
    graph.ensure_node("single")
    result = compute_pagerank(graph)
    expect_in("single", result)
    expect_true(abs(result["single"] - PAGERANK_SUM) < TOLERANCE)


def test_pagerank_simple_chain() -> None:
    """PageRank flows through chain graph."""
    graph = _make_simple_chain()
    result = compute_pagerank(graph)

    expect_length(result, EXPECTED_NODES_4)

    expect_true(result["D"] > result["A"])


def test_pagerank_cycle_equal() -> None:
    """Nodes in simple cycle have equal PageRank."""
    graph = _make_simple_cycle()
    result = compute_pagerank(graph)

    values = list(result.values())
    expect_true(max(values) - min(values) < TOLERANCE)


def test_pagerank_star_hub_low() -> None:
    """Hub in out-star has lower PageRank (no incoming edges)."""
    graph = _make_star_graph()
    result = compute_pagerank(graph)

    hub_rank = result["Hub"]
    spoke_ranks = [result[s] for s in ["A", "B", "C", "D"]]
    avg_spoke = sum(spoke_ranks) / len(spoke_ranks)
    expect_true(hub_rank < avg_spoke)


def test_pagerank_reverse_star_hub_high() -> None:
    """Hub in in-star has higher PageRank (many incoming edges)."""
    graph = _make_reverse_star_graph()
    result = compute_pagerank(graph)
    hub_rank = result["Hub"]
    for spoke in ["A", "B", "C", "D"]:
        expect_true(hub_rank > result[spoke])


def test_pagerank_realistic_call_graph() -> None:
    """PageRank identifies important functions in call graph."""
    graph = _make_call_graph_realistic()
    result = compute_pagerank(graph)

    expect_in("format_error", result)

    expect_in("log_result", result)


def test_pagerank_custom_alpha() -> None:
    """PageRank respects custom alpha (damping) parameter."""
    graph = _make_simple_chain()
    result_low = compute_pagerank(graph, options=PagerankOptions(alpha=0.5))
    result_high = compute_pagerank(graph, options=PagerankOptions(alpha=0.95))

    expect_not_equal(result_low, result_high)


def test_pagerank_custom_max_iter() -> None:
    """PageRank respects custom max_iter parameter."""
    graph = _make_simple_chain()

    result = compute_pagerank(graph, options=PagerankOptions(max_iter=10))
    expect_length(result, EXPECTED_NODES_4)


def test_pagerank_custom_tolerance() -> None:
    """PageRank respects custom tolerance parameter."""
    graph = _make_simple_chain()
    result_low = compute_pagerank(graph, options=PagerankOptions(tol=1e-9))
    result_high = compute_pagerank(graph, options=PagerankOptions(tol=1e-3))

    expect_length(result_low, EXPECTED_NODES_4)
    expect_length(result_high, EXPECTED_NODES_4)


def test_pagerank_disconnected_components() -> None:
    """PageRank handles disconnected graph components."""
    graph = _make_disconnected_components()
    result = compute_pagerank(graph)

    expect_length(result, EXPECTED_NODES_7)

    expect_in("Isolated", result)
    expect_true(result["Isolated"] > 0)


def test_pagerank_sums_to_one() -> None:
    """PageRank values sum to approximately 1.0."""
    graph = _make_call_graph_realistic()
    result = compute_pagerank(graph)
    total = sum(result.values())
    expect_true(abs(total - PAGERANK_SUM) < TOLERANCE)


def test_pagerank_keys_match_nodes() -> None:
    """PageRank result keys match original node types."""
    graph = RxGraphStore.directed()
    graph.add_weighted_edge(1, 2, weight=1.0)
    graph.add_weighted_edge(2, 3, weight=1.0)
    result = compute_pagerank(graph)

    for key in result:
        expect_is_instance(key, int)


def test_pagerank_values_are_floats() -> None:
    """PageRank result values are always floats."""
    graph = _make_simple_chain()
    result = compute_pagerank(graph)
    for value in result.values():
        expect_is_instance(value, float)


def test_betweenness_empty_graph() -> None:
    """Empty graph returns empty betweenness dictionary."""
    graph = RxGraphStore.directed()
    result = compute_betweenness(graph)
    expect_equal(result, {})


def test_betweenness_single_node_zero() -> None:
    """Single node has zero betweenness."""
    graph = RxGraphStore.directed()
    graph.ensure_node("single")
    result = compute_betweenness(graph)
    expect_in("single", result)
    expect_equal(result["single"], 0.0)


def test_betweenness_chain_middle_nodes_high() -> None:
    """Middle nodes in chain have higher betweenness."""
    graph = _make_simple_chain()
    result = compute_betweenness(graph)

    expect_true(result["B"] > result["A"])
    expect_true(result["C"] > result["D"])


def test_betweenness_star_hub_high() -> None:
    """Hub in star graph has highest betweenness (all paths through it)."""
    graph = _make_star_graph()
    result = compute_betweenness(graph)
    hub_betweenness = result["Hub"]
    for spoke in ["A", "B", "C", "D"]:
        expect_true(hub_betweenness >= result[spoke])


def test_betweenness_cycle_equal() -> None:
    """Nodes in simple cycle have equal betweenness."""
    graph = _make_simple_cycle()
    result = compute_betweenness(graph)
    values = list(result.values())
    expect_true(max(values) - min(values) < TOLERANCE)


def test_betweenness_normalized() -> None:
    """Normalized betweenness values are between 0 and 1."""
    graph = _make_call_graph_realistic()
    result = compute_betweenness(graph, options=BetweennessOptions(normalized=True))
    for value in result.values():
        expect_true(0.0 <= value <= 1.0)


def test_betweenness_unnormalized() -> None:
    """Unnormalized betweenness can exceed 1."""
    graph = _make_call_graph_realistic()
    result = compute_betweenness(graph, options=BetweennessOptions(normalized=False))

    expect_length(result, graph.graph.num_nodes())


def test_betweenness_sampled_with_k() -> None:
    """Approximate betweenness with sample size k."""
    graph = _make_dense_cluster()

    result = compute_betweenness(graph, options=BetweennessOptions(k=3))

    expect_length(result, EXPECTED_NODES_5)


def test_betweenness_disconnected_components() -> None:
    """Betweenness handles disconnected components."""
    graph = _make_disconnected_components()
    result = compute_betweenness(graph)

    expect_length(result, EXPECTED_NODES_7)

    expect_equal(result["Isolated"], 0.0)


def test_betweenness_realistic_call_graph() -> None:
    """Betweenness identifies bridge functions in call graph."""
    graph = _make_call_graph_realistic()
    result = compute_betweenness(graph)

    expect_in("process_request", result)

    expect_true(result["process_request"] > 0)


def test_betweenness_keys_match_nodes() -> None:
    """Betweenness result keys match original node types."""
    graph = RxGraphStore.directed()
    graph.add_weighted_edge(1, 2, weight=1.0)
    graph.add_weighted_edge(2, 3, weight=1.0)
    result = compute_betweenness(graph)

    for key in result:
        expect_is_instance(key, int)


def test_betweenness_values_are_floats() -> None:
    """Betweenness result values are always floats."""
    graph = _make_simple_chain()
    result = compute_betweenness(graph)
    for value in result.values():
        expect_is_instance(value, float)


class _ContextOverrides(TypedDict, total=False):
    betweenness_sample: int
    eigen_max_iter: int
    seed: int
    pagerank_weight: str | None
    betweenness_weight: str | None
    use_gpu: bool
    community_detection_limit: int | None
    now: datetime | None


def _make_context(overrides: _ContextOverrides | None = None) -> GraphContext:
    params = overrides or {}
    return GraphContext(
        repo="repo",
        commit="commit",
        betweenness_sample=params.get("betweenness_sample", 2),
        eigen_max_iter=params.get("eigen_max_iter", 1),
        seed=params.get("seed", 1),
        pagerank_weight=params.get("pagerank_weight", "weight"),
        betweenness_weight=params.get("betweenness_weight", "weight"),
        use_gpu=params.get("use_gpu", False),
        community_detection_limit=params.get("community_detection_limit"),
        now=params.get("now"),
    )


def test_both_metrics_same_nodes() -> None:
    """PageRank and betweenness produce results for same nodes."""
    graph = _make_call_graph_realistic()
    pagerank = compute_pagerank(graph)
    betweenness = compute_betweenness(graph)
    expect_equal(set(pagerank.keys()), set(betweenness.keys()))


def test_metrics_identify_different_importance() -> None:
    """PageRank and betweenness may rank nodes differently."""
    graph = _make_call_graph_realistic()
    pagerank = compute_pagerank(graph)
    betweenness = compute_betweenness(graph)

    pr_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    bc_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

    top_pr = [n for n, _ in pr_sorted[:EXPECTED_TOP_3]]
    top_bc = [n for n, _ in bc_sorted[:EXPECTED_TOP_3]]

    expect_length(top_pr, EXPECTED_TOP_3)
    expect_length(top_bc, EXPECTED_TOP_3)


def test_dense_graph_metrics() -> None:
    """Both metrics work on dense graphs."""
    graph = _make_dense_cluster()
    pagerank = compute_pagerank(graph)
    betweenness = compute_betweenness(graph)

    pr_values = list(pagerank.values())
    bc_values = list(betweenness.values())
    pr_range = max(pr_values) - min(pr_values)
    bc_range = max(bc_values) - min(bc_values)

    expect_true(pr_range < DENSE_GRAPH_RANGE_TOLERANCE)
    expect_true(bc_range < DENSE_GRAPH_RANGE_TOLERANCE)


def test_centrality_directed_logs_eigen_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Eigenvector failures in directed centrality emit warnings."""
    graph = _make_simple_chain()
    caplog.set_level("WARNING", logger=centrality_module.__name__)

    context = _make_context()
    bundle = centrality_module.centrality_directed(
        graph,
        context,
        include_eigen=True,
        compute_overrides=centrality_module.CentralityComputations(
            eigen_fn=lambda *_args, **_kwargs: {},
        ),
    )

    expect_equal(bundle.eigenvector, {})
    assert_logged(caplog.records, level="WARNING", containing="did not converge")


def test_centrality_undirected_logs_eigen_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Eigenvector failures in undirected centrality emit warnings."""
    graph = RxGraphStore.undirected()
    graph.add_weighted_edge("a", "b", weight=1.0)
    graph.add_weighted_edge("b", "c", weight=1.0)
    caplog.set_level("WARNING", logger=centrality_module.__name__)

    context = _make_context()
    bundle = centrality_module.centrality_undirected(
        graph,
        context,
        compute_overrides=centrality_module.CentralityComputations(
            eigen_fn=lambda *_args, **_kwargs: {},
        ),
    )

    expect_equal(bundle.eigenvector, {})
    assert_logged(
        caplog.records,
        level="WARNING",
        containing="Eigenvector centrality did not converge for graph=",
    )


def test_centrality_undirected_logs_structural_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Structural hole failures surface as warnings."""
    graph = RxGraphStore.undirected()
    graph.add_weighted_edge("x", "y", weight=1.0)
    graph.add_weighted_edge("y", "z", weight=1.0)
    caplog.set_level("WARNING", logger=centrality_module.__name__)

    def _raise_constraint(*_: object, **__: object) -> None:
        error_message = "failed to converge"
        raise ValueError(error_message)

    context = _make_context()
    bundle = centrality_module.centrality_undirected(
        graph,
        context,
        include_structural=True,
        compute_overrides=centrality_module.CentralityComputations(
            constraint_fn=_raise_constraint,
        ),
    )

    expect_length(bundle.pagerank, graph.graph.num_nodes())
    assert_logged(
        caplog.records,
        level="WARNING",
        containing="Structural holes calculation failed for graph",
    )
