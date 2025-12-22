"""Tests for pure graph metric computation functions.

This module tests the stateless computation functions for centrality,
component analysis, and structural metrics without any database I/O.
"""

from __future__ import annotations

from typing import Final

import networkx as nx
import pytest

from codeintel.core.compute.centrality import (
    CentralityMetrics,
    centrality_to_rows,
    compute_all_centralities,
    compute_betweenness,
    compute_closeness,
    compute_degree_centrality,
    compute_in_degree_centrality,
    compute_out_degree_centrality,
    compute_pagerank,
)
from codeintel.graphs.compute.metrics.components import (
    ComponentInfo,
    SCCResult,
    compute_component_stats,
    condensation_layers,
    find_articulation_points,
    find_bridges,
    find_connected,
    find_cycles,
    find_strongly_connected,
    find_weakly_connected,
    topological_layers,
)
from codeintel.graphs.compute.metrics.structural import (
    StructuralMetrics,
    compute_all_structural,
    compute_constraint,
    compute_core_number,
    compute_effective_size,
    compute_triangles,
)
from codeintel.graphs.compute.metrics.structural import (
    compute_clustering_coefficient as compute_structural_clustering,
)
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_true,
)
from tests._helpers.fakes.networkx_graphs import (
    bridge_chain_graph,
    chain_graph,
    complete_digraph,
    complete_graph,
    complex_sccs_graph,
    cyclic_graph,
    diamond_graph,
    disconnected_graph,
    empty_digraph,
    empty_graph,
    fan_in_fan_out_graph,
    star_graph,
    two_cycle_graph,
    two_sccs_graph,
)
from tests.graphs.constants import CYCLE_SIZE_SWEEP

EXPECTED_CYCLE_NODES: Final[int] = 3
EXPECTED_MIN_COMPONENTS: Final[int] = 2
EXPECTED_SINGLE_COMPONENT: Final[int] = 1
PAGERANK_TOLERANCE: Final[float] = 0.01
INSTABILITY_ZERO: Final[float] = 0.0
INSTABILITY_FULL: Final[float] = 1.0
EXPECTED_NODE_COUNT_TWO: Final[int] = 2
EXPECTED_NODE_COUNT_FOUR: Final[int] = 4
EXPECTED_NODE_COUNT_FIVE: Final[int] = 5
EXPECTED_NODE_COUNT_SIX: Final[int] = 6
EXPECTED_LAYER_TWO: Final[int] = 2
MEAN_SIZE_THREE: Final[float] = 3.0
LARGEST_SIZE_FIVE: Final[int] = 5
INSTABILITY_POINT_FOUR: Final[float] = 0.4
PAGERANK_POINT_FIVE: Final[float] = 0.5
BETWEENNESS_POINT_THREE: Final[float] = 0.3
TRIANGLES_PER_NODE_K4: Final[int] = 3
CORE_NUMBER_K4: Final[int] = 3


def test_pagerank_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = empty_digraph()
    result = compute_pagerank(graph)
    expect_true(result == {})


def test_pagerank_simple_cycle() -> None:
    """Cycle graph has uniform PageRank."""
    graph = cyclic_graph(EXPECTED_CYCLE_NODES)
    result = compute_pagerank(graph)

    expect_true(len(result) == EXPECTED_CYCLE_NODES)

    values = list(result.values())
    expected_uniform = 1.0 / EXPECTED_CYCLE_NODES
    for val in values:
        expect_true(abs(val - expected_uniform) < PAGERANK_TOLERANCE)


def test_pagerank_star_graph_center_has_highest() -> None:
    """Star graph center has highest PageRank."""
    graph = star_graph(4, inward=True)
    result = compute_pagerank(graph)

    expect_true(result["hub"] > result["spoke1"])
    expect_true(result["hub"] > result["spoke2"])


def test_pagerank_custom_alpha() -> None:
    """Custom damping factor works."""
    graph = nx.DiGraph([(1, 2), (2, 3)])
    result_default = compute_pagerank(graph)
    result_low_alpha = compute_pagerank(graph, alpha=0.5)

    expect_true(result_default != result_low_alpha)


def test_pagerank_chain_graph_probability_distribution() -> None:
    """Chain graph PageRank sums to one."""
    graph = chain_graph()
    result = compute_pagerank(graph)

    expect_true(len(result) == graph.number_of_nodes())
    expect_true(abs(sum(result.values()) - 1.0) < PAGERANK_TOLERANCE)


def test_pagerank_single_node_graph() -> None:
    """Single node graph assigns all rank to that node."""
    graph = empty_digraph()
    graph.add_node("solo")

    result = compute_pagerank(graph)

    expect_true(len(result) == EXPECTED_SINGLE_COMPONENT)
    expect_true(abs(result["solo"] - 1.0) < PAGERANK_TOLERANCE)


def test_pagerank_outward_star_hub_has_low_rank() -> None:
    """Outward star distributes rank to spokes."""
    graph = star_graph(3)

    result = compute_pagerank(graph)

    expect_true(result["hub"] < result["spoke1"])
    expect_true(result["hub"] < result["spoke2"])


def test_betweenness_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = empty_digraph()
    result = compute_betweenness(graph)
    expect_true(result == {})


def test_betweenness_path_graph_middle_node_highest() -> None:
    """Middle node in path has highest betweenness."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 5)])
    result = compute_betweenness(graph, normalized=True)

    expect_true(result[3] >= result[1])


def test_betweenness_sampling_parameter() -> None:
    """Sampling parameter k works."""
    graph = nx.DiGraph([(i, i + 1) for i in range(10)])
    result = compute_betweenness(graph, k=3)

    expect_true(len(result) == graph.number_of_nodes())


def test_betweenness_diamond_prioritizes_inner_nodes() -> None:
    """Diamond graph betweenness highlights middle nodes."""
    graph = diamond_graph()
    result = compute_betweenness(graph)

    expect_true(result["B"] >= result["A"])
    expect_true(result["C"] >= result["A"])


def test_betweenness_disconnected_graph_has_entries_for_all_nodes() -> None:
    """Disconnected graph still returns values for all nodes."""
    graph = disconnected_graph()
    result = compute_betweenness(graph)

    expect_true(len(result) == EXPECTED_NODE_COUNT_SIX)


def test_closeness_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = empty_digraph()
    result = compute_closeness(graph)
    expect_true(result == {})


def test_closeness_complete_graph_uniform() -> None:
    """Complete graph has uniform closeness."""
    graph = complete_graph(5)
    result = compute_closeness(graph)

    values = list(result.values())

    expect_true(all(abs(v - values[0]) < PAGERANK_TOLERANCE for v in values))


def test_closeness_wf_improved_parameter() -> None:
    """Wasserman-Faust improvement parameter works."""
    graph = nx.path_graph(5)
    result_improved = compute_closeness(graph, wf_improved=True)
    result_basic = compute_closeness(graph, wf_improved=False)

    expect_true(len(result_improved) == len(result_basic))


def test_closeness_disconnected_graph_returns_all_nodes() -> None:
    """Disconnected graph returns closeness for each node."""
    graph = disconnected_graph()
    result = compute_closeness(graph)

    expect_true(len(result) == EXPECTED_NODE_COUNT_SIX)


def test_degree_centrality_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = empty_digraph()
    result = compute_degree_centrality(graph)
    expect_true(result == {})


def test_in_degree_centrality() -> None:
    """In-degree centrality computation."""
    graph = fan_in_fan_out_graph(sources=("s1", "s2", "s3"), sinks=("t1",))
    result = compute_in_degree_centrality(graph)

    expect_true(result["core"] > result["s1"])


def test_in_degree_centrality_empty_graph() -> None:
    """Empty graph returns empty in-degree centrality."""
    graph = empty_digraph()
    result = compute_in_degree_centrality(graph)

    expect_true(result == {})


def test_out_degree_centrality() -> None:
    """Out-degree centrality computation."""
    graph = fan_in_fan_out_graph(sinks=("out1", "out2", "out3"))
    result = compute_out_degree_centrality(graph)

    expect_true(result["core"] > result["out1"])


def test_out_degree_centrality_empty_graph() -> None:
    """Empty graph returns empty out-degree centrality."""
    graph = empty_digraph()
    result = compute_out_degree_centrality(graph)

    expect_true(result == {})


def test_all_centralities_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = empty_digraph()
    result = compute_all_centralities(graph)
    expect_true(result == {})


def test_all_centralities_single_node_returns_zero_degrees() -> None:
    """Single node graph returns zero degrees."""
    graph = empty_digraph()
    graph.add_node("solo")
    result = compute_all_centralities(graph)

    expect_true(len(result) == EXPECTED_SINGLE_COMPONENT)
    metrics = result["solo"]
    expect_true(metrics.in_degree == 0)
    expect_true(metrics.out_degree == 0)
    expect_true(metrics.degree == 0)


def test_all_centralities_returns_dataclass() -> None:
    """Returns CentralityMetrics dataclass for each node."""
    graph = nx.DiGraph([(1, 2), (2, 3)])
    result = compute_all_centralities(graph)

    expect_true(len(result) == EXPECTED_CYCLE_NODES)
    for metrics in result.values():
        expect_true(isinstance(metrics, CentralityMetrics))
        expect_true(hasattr(metrics, "pagerank"))
        expect_true(hasattr(metrics, "betweenness"))
        expect_true(hasattr(metrics, "closeness"))
        expect_true(hasattr(metrics, "in_degree"))
        expect_true(hasattr(metrics, "out_degree"))
        expect_true(hasattr(metrics, "degree"))


def test_all_centralities_degree_calculation() -> None:
    """Degree calculation is sum of in and out degree."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 2)])
    result = compute_all_centralities(graph)

    for metrics in result.values():
        expected_degree = metrics.in_degree + metrics.out_degree
        expect_true(metrics.degree == expected_degree)


def test_centrality_to_rows_converts_metrics() -> None:
    """Converts CentralityMetrics to row dicts."""
    metrics = {
        1: CentralityMetrics(
            pagerank=PAGERANK_POINT_FIVE,
            betweenness=BETWEENNESS_POINT_THREE,
            closeness=INSTABILITY_POINT_FOUR,
            harmonic=PAGERANK_POINT_FIVE,
            eigenvector=BETWEENNESS_POINT_THREE,
            in_degree=2,
            out_degree=1,
            degree=3,
        )
    }
    rows = centrality_to_rows(metrics, repo="test-repo", commit="abc123")

    expect_true(len(rows) == 1)
    row = rows[0]
    expect_true(row["goid_h128"] == 1)
    expect_true(row["repo"] == "test-repo")
    expect_true(row["commit"] == "abc123")
    expect_true(row["pagerank"] == PAGERANK_POINT_FIVE)
    expect_true(row["betweenness"] == BETWEENNESS_POINT_THREE)


def test_scc_empty_graph_returns_empty() -> None:
    """Empty graph returns empty result."""
    graph = empty_digraph()
    result = find_strongly_connected(graph)

    expect_true(result.components == ())
    expect_true(result.node_to_component == {})
    expect_true(result.condensation is None)


@pytest.mark.parametrize("cycle_size", CYCLE_SIZE_SWEEP)
def test_scc_simple_cycle_is_one_scc(cycle_size: int) -> None:
    """Simple cycles are single SCCs."""
    graph = cyclic_graph(cycle_size)
    result = find_strongly_connected(graph)

    expect_true(len(result.components) == EXPECTED_SINGLE_COMPONENT)
    expect_true(result.components[0].size == cycle_size)


def test_scc_disconnected_nodes_are_separate() -> None:
    """Disconnected nodes are separate SCCs."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (4, 5)])
    result = find_strongly_connected(graph)

    expect_true(len(result.components) >= EXPECTED_MIN_COMPONENTS)


def test_scc_dag_nodes_are_singletons() -> None:
    """DAG nodes are individual SCCs."""
    graph = chain_graph()
    result = find_strongly_connected(graph)

    expect_true(len(result.components) == graph.number_of_nodes())
    expect_true(all(comp.size == 1 for comp in result.components))


def test_scc_node_to_component_mapping() -> None:
    """Node to component mapping is correct."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    result = find_strongly_connected(graph)

    comp_id = result.node_to_component[1]
    expect_true(result.node_to_component[2] == comp_id)
    expect_true(result.node_to_component[3] == comp_id)


def test_scc_condensation_graph_computed() -> None:
    """Condensation graph computed when requested."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (1, 4)])
    result = find_strongly_connected(graph, compute_condensation=True)

    expect_true(result.condensation is not None)
    expect_true(isinstance(result.condensation, nx.DiGraph))


def test_scc_two_components_sizes() -> None:
    """Two SCC graph returns expected component sizes."""
    graph = two_sccs_graph()
    result = find_strongly_connected(graph)

    sizes = sorted(comp.size for comp in result.components)
    expect_true(len(sizes) == EXPECTED_MIN_COMPONENTS)
    expect_equal(sizes[0], EXPECTED_NODE_COUNT_TWO)
    expect_equal(sizes[1], EXPECTED_NODE_COUNT_TWO)


def test_scc_complex_component_mix() -> None:
    """Complex SCC graph finds all component sizes."""
    graph = complex_sccs_graph()
    result = find_strongly_connected(graph)

    sizes = sorted(comp.size for comp in result.components)
    expect_equal(sizes, [1, 2, 3])


def test_scc_single_node_component() -> None:
    """Single node graph returns one SCC."""
    graph = empty_digraph()
    graph.add_node("solo")
    result = find_strongly_connected(graph)

    expect_true(len(result.components) == EXPECTED_SINGLE_COMPONENT)
    expect_true(result.components[0].size == EXPECTED_SINGLE_COMPONENT)


def test_wcc_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = empty_digraph()
    result = find_weakly_connected(graph)
    expect_true(result == [])


def test_wcc_connected_graph_is_one_wcc() -> None:
    """Connected graph is one WCC."""
    graph = nx.DiGraph([(1, 2), (2, 3), (4, 3)])
    result = find_weakly_connected(graph)

    expect_true(len(result) == EXPECTED_SINGLE_COMPONENT)


def test_wcc_component_info_structure() -> None:
    """ComponentInfo has correct structure."""
    graph = nx.DiGraph([(1, 2)])
    result = find_weakly_connected(graph)

    expect_true(len(result) == EXPECTED_SINGLE_COMPONENT)
    comp = result[0]
    expect_true(isinstance(comp, ComponentInfo))
    expect_true(comp.component_id == 0)
    expect_true(comp.size == EXPECTED_NODE_COUNT_TWO)
    expect_true(1 in comp.nodes)
    expect_true(EXPECTED_NODE_COUNT_TWO in comp.nodes)


def test_connected_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = empty_graph()
    result = find_connected(graph)
    expect_true(result == [])


def test_connected_graph_is_one_component() -> None:
    """Connected graph is one component."""
    graph = nx.Graph([(1, 2), (2, 3)])
    result = find_connected(graph)

    expect_true(len(result) == EXPECTED_SINGLE_COMPONENT)


def test_bridges_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = empty_graph()
    result = find_bridges(graph)
    expect_true(result == [])


def test_bridges_path_graph_all_edges_are_bridges() -> None:
    """Path graph - all edges are bridges."""
    graph = bridge_chain_graph(segments=4, segment_size=1)
    result = find_bridges(graph)

    expect_equal(len(result), EXPECTED_CYCLE_NODES)


def test_bridges_cycle_has_no_bridges() -> None:
    """Cycle has no bridges."""
    graph = nx.cycle_graph(4)
    result = find_bridges(graph)

    expect_true(len(result) == 0)


def test_articulation_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = empty_graph()
    result = find_articulation_points(graph)
    expect_true(result == [])


def test_articulation_path_graph_middle_nodes() -> None:
    """Path graph - middle nodes are articulation points."""
    graph = nx.path_graph(5)
    result = find_articulation_points(graph)

    expect_true(len(result) == EXPECTED_CYCLE_NODES)


def test_articulation_complete_graph_no_articulation() -> None:
    """Complete graph has no articulation points."""
    graph = complete_graph(5)
    result = find_articulation_points(graph)

    expect_true(len(result) == 0)


def test_component_stats_empty_returns_zeros() -> None:
    """Empty components returns zero stats."""
    result = compute_component_stats([])

    expect_true(result["count"] == 0)
    expect_true(result["largest_size"] == 0)
    expect_true(result["smallest_size"] == 0)
    expect_true(result["mean_size"] == 0.0)
    expect_true(result["singleton_count"] == 0)


def test_component_stats_computes_correct() -> None:
    """Computes correct statistics."""
    components = [
        ComponentInfo(component_id=0, size=5, nodes=frozenset(range(5))),
        ComponentInfo(component_id=1, size=1, nodes=frozenset([10])),
        ComponentInfo(component_id=2, size=3, nodes=frozenset([20, 21, 22])),
    ]
    result = compute_component_stats(components)

    expect_true(result["count"] == EXPECTED_CYCLE_NODES)
    expect_true(result["largest_size"] == LARGEST_SIZE_FIVE)
    expect_true(result["smallest_size"] == 1)
    expect_true(result["mean_size"] == MEAN_SIZE_THREE)
    expect_true(result["singleton_count"] == 1)


def test_cycles_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = empty_digraph()
    result = find_cycles(graph)
    expect_true(result == [])


@pytest.mark.parametrize("cycle_size", CYCLE_SIZE_SWEEP)
def test_cycles_simple_cycle_detected(cycle_size: int) -> None:
    """Simple cycle is detected."""
    graph = cyclic_graph(cycle_size)
    result = find_cycles(graph)

    expect_true(len(result) >= 1)
    cycle_nodes = set(result[0])
    expect_equal(len(cycle_nodes), cycle_size)


def test_cycles_limit_parameter_respected() -> None:
    """Limit parameter is respected."""
    graph = nx.relabel_nodes(
        two_cycle_graph(),
        mapping={"A": 1, "B": 2, "C": 3, "D": 4},
    )
    result = find_cycles(graph, limit=1)

    expect_true(len(result) <= 1)


def test_cycles_dag_has_no_cycles() -> None:
    """DAG has no cycles."""
    graph = nx.DiGraph([(1, 2), (2, 3), (1, 3)])
    result = find_cycles(graph)

    expect_true(len(result) == 0)


def test_topological_layers_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = empty_digraph()
    result = topological_layers(graph)
    expect_true(result == {})


def test_topological_layers_chain_graph() -> None:
    """Chain graph has incremental layers."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    result = topological_layers(graph)

    expect_true(result[1] == 0)
    expect_true(result[2] == 1)
    expect_true(result[3] == EXPECTED_LAYER_TWO)
    expect_true(result[4] == EXPECTED_CYCLE_NODES)


def test_topological_layers_root_nodes_layer_zero() -> None:
    """Root nodes have layer 0."""
    graph = nx.DiGraph([(1, 3), (2, 3)])
    result = topological_layers(graph)

    expect_true(result[1] == 0)
    expect_true(result[2] == 0)
    expect_true(result[3] == 1)


def test_condensation_layers_no_condensation_returns_empty() -> None:
    """No condensation returns empty dict."""
    scc_result = SCCResult(components=(), node_to_component={}, condensation=None)
    graph = empty_digraph()
    result = condensation_layers(graph, scc_result)

    expect_true(result == {})


def test_condensation_layers_with_condensation() -> None:
    """With condensation computes layers."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (1, 4)])
    scc_result = find_strongly_connected(graph, compute_condensation=True)
    result = condensation_layers(graph, scc_result)

    expect_true(len(result) == graph.number_of_nodes())


def test_condensation_layers_respects_component_order() -> None:
    """Condensation layers order SCCs topologically."""
    graph = two_sccs_graph()
    scc_result = find_strongly_connected(graph, compute_condensation=True)
    layers = condensation_layers(graph, scc_result)

    expect_equal(layers["A"], layers["B"])
    expect_equal(layers["C"], layers["D"])
    expect_true(layers["C"] > layers["A"])


def test_structural_clustering_handles_directed_graphs() -> None:
    """Structural clustering converts directed graphs."""
    graph = complete_digraph(EXPECTED_NODE_COUNT_FOUR)
    result = compute_structural_clustering(graph)

    expect_true(len(result) == EXPECTED_NODE_COUNT_FOUR)


def test_structural_triangles_complete_graph_counts() -> None:
    """Complete graphs have predictable triangle counts."""
    graph = complete_graph(EXPECTED_NODE_COUNT_FOUR)
    result = compute_triangles(graph)

    expect_true(all(count == TRIANGLES_PER_NODE_K4 for count in result.values()))


def test_structural_core_number_chain_graph() -> None:
    """Chain graph yields core number of 1 for all nodes."""
    graph = chain_graph().to_undirected()
    result = compute_core_number(graph)

    expect_true(all(core == EXPECTED_SINGLE_COMPONENT for core in result.values()))


def test_structural_constraint_single_node_zero() -> None:
    """Constraint for isolated node is zero."""
    graph = empty_graph()
    graph.add_node("solo")
    result = compute_constraint(graph)

    expect_true(result["solo"] == INSTABILITY_ZERO)


def test_structural_effective_size_star_hub_larger_than_spokes() -> None:
    """Effective size favors hub in a star graph."""
    graph = star_graph(4).to_undirected()
    result = compute_effective_size(graph)

    expect_true(result["hub"] > result["spoke1"])


def test_all_structural_complete_graph_metrics() -> None:
    """All structural metrics are populated for complete graph."""
    graph = complete_graph(EXPECTED_NODE_COUNT_FOUR)
    result = compute_all_structural(graph)

    expect_true(len(result) == EXPECTED_NODE_COUNT_FOUR)
    for metrics in result.values():
        expect_true(isinstance(metrics, StructuralMetrics))
        expect_true(abs(metrics.clustering - INSTABILITY_FULL) < PAGERANK_TOLERANCE)
        expect_true(metrics.triangles == TRIANGLES_PER_NODE_K4)
        expect_true(metrics.core_number == CORE_NUMBER_K4)


def test_all_structural_empty_graph_returns_empty() -> None:
    """Empty graph returns empty structural metrics."""
    graph = empty_graph()
    result = compute_all_structural(graph)

    expect_true(result == {})


def test_centrality_metrics_frozen() -> None:
    """CentralityMetrics is frozen."""
    metrics = CentralityMetrics(
        pagerank=PAGERANK_POINT_FIVE,
        betweenness=BETWEENNESS_POINT_THREE,
        closeness=INSTABILITY_POINT_FOUR,
        harmonic=PAGERANK_POINT_FIVE,
        eigenvector=BETWEENNESS_POINT_THREE,
        in_degree=2,
        out_degree=1,
        degree=3,
    )
    assert_cannot_setattr(metrics, "pagerank", 0.9)


def test_component_info_frozen() -> None:
    """ComponentInfo is frozen."""
    info = ComponentInfo(component_id=0, size=5, nodes=frozenset([1, 2, 3, 4, 5]))
    assert_cannot_setattr(info, "size", 10)


def test_scc_result_frozen() -> None:
    """SCCResult is frozen."""
    result = SCCResult(components=(), node_to_component={})
    assert_cannot_setattr(result, "condensation", empty_digraph())


GOLDEN_MIN_NODES: Final[int] = 13
GOLDEN_EXPECTED_SCC: Final[int] = 1


def _build_realistic_call_graph() -> nx.DiGraph:
    """Build a realistic call graph simulating production patterns.

    Returns
    -------
    nx.DiGraph
        A directed graph with hub functions, layered architecture, and SCCs.
    """
    g = empty_digraph()

    core_funcs = ["format_string", "parse_json", "validate_input", "hash_value"]
    g.add_nodes_from(core_funcs)

    services = ["authenticate", "query", "execute", "get_cached", "set_cached"]
    g.add_nodes_from(services)
    for s in services:
        g.add_edge(s, "validate_input")
        g.add_edge(s, "format_string")

    handlers = ["create_user", "get_user", "update_user", "delete_user", "create_order"]
    g.add_nodes_from(handlers)
    for h in handlers:
        g.add_edge(h, "authenticate")
        g.add_edge(h, "query")
        g.add_edge(h, "get_cached")

    api = ["handle_request", "register_routes"]
    g.add_nodes_from(api)
    for a in api:
        for h in handlers:
            g.add_edge(a, h)

    g.add_node("log_info")
    for node in services + handlers:
        g.add_edge(node, "log_info")

    g.add_edge("authenticate", "get_cached")
    g.add_edge("get_cached", "authenticate")

    return g


def _build_realistic_import_graph() -> nx.DiGraph:
    """Build a realistic import graph with layered architecture.

    Returns
    -------
    nx.DiGraph
        A directed graph representing module imports.
    """
    g = empty_digraph()

    core = ["core.utils", "core.types", "core.errors", "core.config"]
    g.add_nodes_from(core)

    services = ["services.auth", "services.cache", "services.database"]
    g.add_nodes_from(services)
    for s in services:
        g.add_edge(s, "core.utils")
        g.add_edge(s, "core.errors")

    handlers = ["handlers.user", "handlers.product", "handlers.order"]
    g.add_nodes_from(handlers)
    for h in handlers:
        g.add_edge(h, "services.auth")
        g.add_edge(h, "services.database")
        g.add_edge(h, "core.errors")

    api = ["api.routes", "api.middleware"]
    g.add_nodes_from(api)
    for a in api:
        for h in handlers:
            g.add_edge(a, h)

    g.add_node("utils.logging")
    for node in services + handlers + api:
        g.add_edge(node, "utils.logging")

    g.add_edge("services.auth", "services.cache")
    g.add_edge("services.cache", "services.auth")

    return g


def test_realistic_pagerank_identifies_hub_functions() -> None:
    """PageRank correctly identifies hub functions in realistic graphs."""
    graph = _build_realistic_call_graph()

    result = compute_pagerank(graph)

    hub_rank = result.get("log_info", 0)

    expect_true(hub_rank > 0)

    expect_true(len(result) >= GOLDEN_MIN_NODES)


def test_realistic_scc_finds_cycles() -> None:
    """SCC detection correctly identifies cycles in realistic graphs."""
    graph = _build_realistic_call_graph()

    result = find_strongly_connected(graph)

    non_trivial_sccs = [comp for comp in result.components if comp.size > 1]
    expect_true(len(non_trivial_sccs) >= GOLDEN_EXPECTED_SCC)


def test_realistic_import_layers_computed() -> None:
    """Topological layers work on realistic import graphs."""
    graph = _build_realistic_import_graph()

    scc_result = find_strongly_connected(graph, compute_condensation=True)

    layers = condensation_layers(graph, scc_result)

    expect_true(len(layers) >= EXPECTED_NODE_COUNT_TWO)


def test_realistic_centrality_metrics() -> None:
    """All centrality metrics work on realistic graphs."""
    graph = _build_realistic_call_graph()

    metrics = compute_all_centralities(graph)

    expect_true(len(metrics) >= GOLDEN_MIN_NODES)

    for metric in metrics.values():
        expect_true(metric.pagerank >= 0)
        expect_true(metric.betweenness >= 0)
        expect_true(metric.in_degree >= 0)
        expect_true(metric.out_degree >= 0)


def test_realistic_component_stats() -> None:
    """Component statistics work on realistic graphs."""
    graph = _build_realistic_import_graph()

    sccs = find_strongly_connected(graph)
    stats = compute_component_stats(sccs.components)

    expect_true(stats["count"] >= GOLDEN_EXPECTED_SCC)
    expect_true(stats["mean_size"] > 0)
    expect_true(stats["largest_size"] >= 1)
