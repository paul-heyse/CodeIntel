"""Tests for pure graph metric computation functions.

This module tests the stateless computation functions for centrality,
component analysis, and coupling metrics without any database I/O.
"""

from __future__ import annotations

from typing import Final

import networkx as nx
import pytest

from codeintel.graphs.compute.metrics.centrality import (
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
from codeintel.graphs.compute.metrics.coupling import (
    Community,
    CouplingMetrics,
    compute_abstractness,
    compute_average_clustering,
    compute_clustering_coefficient,
    compute_coupling,
    compute_distance_from_main_sequence,
    compute_modularity,
    coupling_to_rows,
    detect_communities_label_propagation,
    detect_communities_louvain,
    find_boundary_nodes,
    find_hub_nodes,
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
    expect_is_not_none,
    expect_true,
)
from tests._helpers.fakes.networkx_graphs import (
    bidirectional_deps_graph,
    chain_graph,
    complete_digraph,
    complete_graph,
    complex_sccs_graph,
    cyclic_graph,
    diamond_graph,
    disconnected_graph,
    god_module_graph,
    hub_dependencies_graph,
    independent_modules_graph,
    linear_dependency_graph,
    star_graph,
    two_sccs_graph,
)

# ---------------------------------------------------------------------------
# Constants for magic value compliance
# ---------------------------------------------------------------------------
EXPECTED_CYCLE_NODES: Final[int] = 3
EXPECTED_MIN_COMPONENTS: Final[int] = 2
EXPECTED_SINGLE_COMPONENT: Final[int] = 1
PAGERANK_TOLERANCE: Final[float] = 0.01
INSTABILITY_HALF: Final[float] = 0.5
INSTABILITY_ZERO: Final[float] = 0.0
INSTABILITY_FULL: Final[float] = 1.0
MODULARITY_MIN: Final[float] = -0.5
MODULARITY_MAX: Final[float] = 1.0
MIN_HUB_DEGREE: Final[int] = 5
HUB_THRESHOLD_RATIO: Final[float] = 0.1
EXPECTED_NODE_COUNT_TWO: Final[int] = 2
EXPECTED_NODE_COUNT_FOUR: Final[int] = 4
EXPECTED_NODE_COUNT_FIVE: Final[int] = 5
EXPECTED_NODE_COUNT_SIX: Final[int] = 6
EXPECTED_LAYER_TWO: Final[int] = 2
EXPECTED_EFFERENT_TWO: Final[int] = 2
EXPECTED_INSTABILITY_TWO_THIRDS: Final[float] = 2 / 3
MEAN_SIZE_THREE: Final[float] = 3.0
LARGEST_SIZE_FIVE: Final[int] = 5
AFFERENT_THREE: Final[int] = 3
EFFERENT_TWO: Final[int] = 2
INSTABILITY_POINT_FOUR: Final[float] = 0.4
PAGERANK_POINT_FIVE: Final[float] = 0.5
BETWEENNESS_POINT_THREE: Final[float] = 0.3
STAR_GRAPH_SIZE_TEN: Final[int] = 10
COMMUNITY_SIZE_THREE: Final[int] = 3
HUB_DEPENDENT_COUNT: Final[int] = 4
TRIANGLES_PER_NODE_K4: Final[int] = 3
CORE_NUMBER_K4: Final[int] = 3


# ===========================================================================
# CENTRALITY TESTS
# ===========================================================================


def test_pagerank_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
    result = compute_pagerank(graph)
    expect_true(result == {})


def test_pagerank_simple_cycle() -> None:
    """Cycle graph has uniform PageRank."""
    graph = cyclic_graph(EXPECTED_CYCLE_NODES)
    result = compute_pagerank(graph)

    expect_true(len(result) == EXPECTED_CYCLE_NODES)
    # In a cycle, all nodes should have similar PageRank
    values = list(result.values())
    expected_uniform = 1.0 / EXPECTED_CYCLE_NODES
    for val in values:
        expect_true(abs(val - expected_uniform) < PAGERANK_TOLERANCE)


def test_pagerank_star_graph_center_has_highest() -> None:
    """Star graph center has highest PageRank."""
    graph = star_graph(4, inward=True)
    result = compute_pagerank(graph)

    # Center should have highest PageRank
    expect_true(result["hub"] > result["spoke1"])
    expect_true(result["hub"] > result["spoke2"])


def test_pagerank_custom_alpha() -> None:
    """Custom damping factor works."""
    graph = nx.DiGraph([(1, 2), (2, 3)])
    result_default = compute_pagerank(graph)
    result_low_alpha = compute_pagerank(graph, alpha=0.5)

    # Results should differ with different alpha
    expect_true(result_default != result_low_alpha)


def test_pagerank_chain_graph_probability_distribution() -> None:
    """Chain graph PageRank sums to one."""
    graph = chain_graph()
    result = compute_pagerank(graph)

    expect_true(len(result) == graph.number_of_nodes())
    expect_true(abs(sum(result.values()) - 1.0) < PAGERANK_TOLERANCE)


def test_pagerank_single_node_graph() -> None:
    """Single node graph assigns all rank to that node."""
    graph = nx.DiGraph()
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
    graph = nx.DiGraph()
    result = compute_betweenness(graph)
    expect_true(result == {})


def test_betweenness_path_graph_middle_node_highest() -> None:
    """Middle node in path has highest betweenness."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 5)])
    result = compute_betweenness(graph, normalized=True)

    # Nodes 2, 3, 4 are on paths between others
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
    graph = nx.DiGraph()
    result = compute_closeness(graph)
    expect_true(result == {})


def test_closeness_complete_graph_uniform() -> None:
    """Complete graph has uniform closeness."""
    graph = complete_graph(5)
    result = compute_closeness(graph)

    values = list(result.values())
    # All nodes should have same closeness in complete graph
    expect_true(all(abs(v - values[0]) < PAGERANK_TOLERANCE for v in values))


def test_closeness_wf_improved_parameter() -> None:
    """Wasserman-Faust improvement parameter works."""
    graph = nx.path_graph(5)
    result_improved = compute_closeness(graph, wf_improved=True)
    result_basic = compute_closeness(graph, wf_improved=False)

    # Both should return valid results
    expect_true(len(result_improved) == len(result_basic))


def test_closeness_disconnected_graph_returns_all_nodes() -> None:
    """Disconnected graph returns closeness for each node."""
    graph = disconnected_graph()
    result = compute_closeness(graph)

    expect_true(len(result) == EXPECTED_NODE_COUNT_SIX)


def test_degree_centrality_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
    result = compute_degree_centrality(graph)
    expect_true(result == {})


def test_in_degree_centrality() -> None:
    """In-degree centrality computation."""
    graph = nx.DiGraph([(1, 0), (2, 0), (3, 0)])
    result = compute_in_degree_centrality(graph)

    # Node 0 has highest in-degree
    expect_true(result[0] > result[1])


def test_in_degree_centrality_empty_graph() -> None:
    """Empty graph returns empty in-degree centrality."""
    graph = nx.DiGraph()
    result = compute_in_degree_centrality(graph)

    expect_true(result == {})


def test_out_degree_centrality() -> None:
    """Out-degree centrality computation."""
    graph = nx.DiGraph([(0, 1), (0, 2), (0, 3)])
    result = compute_out_degree_centrality(graph)

    # Node 0 has highest out-degree
    expect_true(result[0] > result[1])


def test_out_degree_centrality_empty_graph() -> None:
    """Empty graph returns empty out-degree centrality."""
    graph = nx.DiGraph()
    result = compute_out_degree_centrality(graph)

    expect_true(result == {})


def test_all_centralities_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
    result = compute_all_centralities(graph)
    expect_true(result == {})


def test_all_centralities_single_node_returns_zero_degrees() -> None:
    """Single node graph returns zero degrees."""
    graph = nx.DiGraph()
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


# ===========================================================================
# COMPONENT TESTS
# ===========================================================================


def test_scc_empty_graph_returns_empty() -> None:
    """Empty graph returns empty result."""
    graph = nx.DiGraph()
    result = find_strongly_connected(graph)

    expect_true(result.components == ())
    expect_true(result.node_to_component == {})
    expect_true(result.condensation is None)


def test_scc_simple_cycle_is_one_scc() -> None:
    """Simple cycle is one SCC."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    result = find_strongly_connected(graph)

    expect_true(len(result.components) == EXPECTED_SINGLE_COMPONENT)
    expect_true(result.components[0].size == EXPECTED_CYCLE_NODES)


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

    # All nodes in same component should have same ID
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
    graph = nx.DiGraph()
    graph.add_node("solo")
    result = find_strongly_connected(graph)

    expect_true(len(result.components) == EXPECTED_SINGLE_COMPONENT)
    expect_true(result.components[0].size == EXPECTED_SINGLE_COMPONENT)


def test_wcc_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = nx.DiGraph()
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
    graph = nx.Graph()
    result = find_connected(graph)
    expect_true(result == [])


def test_connected_graph_is_one_component() -> None:
    """Connected graph is one component."""
    graph = nx.Graph([(1, 2), (2, 3)])
    result = find_connected(graph)

    expect_true(len(result) == EXPECTED_SINGLE_COMPONENT)


def test_bridges_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = nx.Graph()
    result = find_bridges(graph)
    expect_true(result == [])


def test_bridges_path_graph_all_edges_are_bridges() -> None:
    """Path graph - all edges are bridges."""
    graph = nx.path_graph(4)
    result = find_bridges(graph)

    # All edges in a path are bridges
    expect_equal(len(result), EXPECTED_CYCLE_NODES)  # 3 edges for 4 nodes


def test_bridges_cycle_has_no_bridges() -> None:
    """Cycle has no bridges."""
    graph = nx.cycle_graph(4)
    result = find_bridges(graph)

    expect_true(len(result) == 0)


def test_articulation_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = nx.Graph()
    result = find_articulation_points(graph)
    expect_true(result == [])


def test_articulation_path_graph_middle_nodes() -> None:
    """Path graph - middle nodes are articulation points."""
    graph = nx.path_graph(5)
    result = find_articulation_points(graph)

    # Nodes 1, 2, 3 are articulation points
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
    graph = nx.DiGraph()
    result = find_cycles(graph)
    expect_true(result == [])


def test_cycles_simple_cycle_detected() -> None:
    """Simple cycle is detected."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    result = find_cycles(graph)

    expect_true(len(result) >= 1)
    # The cycle contains nodes 1, 2, 3
    cycle_nodes = set(result[0])
    expect_true(
        1 in cycle_nodes
        or EXPECTED_NODE_COUNT_TWO in cycle_nodes
        or EXPECTED_CYCLE_NODES in cycle_nodes
    )


def test_cycles_limit_parameter_respected() -> None:
    """Limit parameter is respected."""
    # Graph with multiple cycles
    graph = nx.DiGraph([(1, 2), (2, 1), (3, 4), (4, 3)])
    result = find_cycles(graph, limit=1)

    expect_true(len(result) <= 1)


def test_cycles_dag_has_no_cycles() -> None:
    """DAG has no cycles."""
    graph = nx.DiGraph([(1, 2), (2, 3), (1, 3)])
    result = find_cycles(graph)

    expect_true(len(result) == 0)


def test_topological_layers_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
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
    graph = nx.DiGraph()
    result = condensation_layers(graph, scc_result)

    expect_true(result == {})


def test_condensation_layers_with_condensation() -> None:
    """With condensation computes layers."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (1, 4)])
    scc_result = find_strongly_connected(graph, compute_condensation=True)
    result = condensation_layers(graph, scc_result)

    # All nodes should have layers assigned
    expect_true(len(result) == graph.number_of_nodes())


def test_condensation_layers_respects_component_order() -> None:
    """Condensation layers order SCCs topologically."""
    graph = two_sccs_graph()
    scc_result = find_strongly_connected(graph, compute_condensation=True)
    layers = condensation_layers(graph, scc_result)

    expect_equal(layers["A"], layers["B"])
    expect_equal(layers["C"], layers["D"])
    expect_true(layers["C"] > layers["A"])


# ===========================================================================
# COUPLING TESTS
# ===========================================================================


def test_coupling_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.DiGraph()
    result = compute_coupling(graph)
    expect_true(result == {})


def test_coupling_computes_afferent_efferent() -> None:
    """Computes afferent and efferent coupling."""
    # Node 1 has 2 outgoing, 1 incoming
    graph = nx.DiGraph([(1, 2), (1, 3), (4, 1)])
    result = compute_coupling(graph)

    expect_true(result[1].efferent == EXPECTED_EFFERENT_TWO)
    expect_true(result[1].afferent == 1)


def test_coupling_instability_calculation() -> None:
    """Instability is efferent / total."""
    graph = nx.DiGraph([(1, 2), (1, 3), (4, 1)])
    result = compute_coupling(graph)

    # Node 1: efferent=2, afferent=1, instability = 2/3
    expect_true(abs(result[1].instability - EXPECTED_INSTABILITY_TWO_THIRDS) < PAGERANK_TOLERANCE)


def test_coupling_isolated_node_zero_instability() -> None:
    """Isolated node has zero instability."""
    graph = nx.DiGraph()
    graph.add_node(1)
    result = compute_coupling(graph)

    expect_true(result[1].instability == INSTABILITY_ZERO)


def test_coupling_sink_node_zero_instability() -> None:
    """Sink node (only incoming) has zero instability."""
    graph = nx.DiGraph([(1, 2), (3, 2)])
    result = compute_coupling(graph)

    expect_true(result[2].instability == INSTABILITY_ZERO)


def test_coupling_source_node_full_instability() -> None:
    """Source node (only outgoing) has instability 1.0."""
    graph = nx.DiGraph([(1, 2), (1, 3)])
    result = compute_coupling(graph)

    expect_true(result[1].instability == INSTABILITY_FULL)


def test_coupling_independent_modules_zero_coupling() -> None:
    """Independent modules have zero afferent/efferent."""
    graph = independent_modules_graph()
    result = compute_coupling(graph)

    expect_true(all(metrics.afferent == 0 for metrics in result.values()))
    expect_true(all(metrics.efferent == 0 for metrics in result.values()))
    expect_true(all(metrics.instability == INSTABILITY_ZERO for metrics in result.values()))


def test_coupling_linear_dependencies_match_directions() -> None:
    """Linear dependency chain yields graded instability."""
    graph = linear_dependency_graph()
    result = compute_coupling(graph)

    expect_equal(result["module_a"].instability, INSTABILITY_FULL)
    expect_true(abs(result["module_b"].instability - INSTABILITY_HALF) < PAGERANK_TOLERANCE)
    expect_equal(result["module_c"].instability, INSTABILITY_ZERO)


def test_coupling_hub_dependencies_concentrate_afferent() -> None:
    """Hub dependencies concentrate afferent coupling on core."""
    graph = hub_dependencies_graph()
    result = compute_coupling(graph)

    expect_equal(result["core"].afferent, HUB_DEPENDENT_COUNT)
    expect_equal(result["core"].instability, INSTABILITY_ZERO)
    expect_equal(result["module_a"].efferent, EXPECTED_SINGLE_COMPONENT)
    expect_equal(result["module_b"].efferent, EXPECTED_SINGLE_COMPONENT)


def test_coupling_god_module_concentrates_efferent() -> None:
    """God module pushes instability outward."""
    graph = god_module_graph()
    result = compute_coupling(graph)

    expect_equal(result["god"].efferent, HUB_DEPENDENT_COUNT)
    expect_equal(result["god"].afferent, 0)
    expect_equal(result["god"].instability, INSTABILITY_FULL)
    expect_equal(result["module_a"].instability, INSTABILITY_ZERO)


def test_coupling_bidirectional_pair_balances_instability() -> None:
    """Bidirectional dependencies share balanced instability."""
    graph = bidirectional_deps_graph()
    result = compute_coupling(graph)

    expect_equal(result["module_a"].afferent, EXPECTED_SINGLE_COMPONENT)
    expect_equal(result["module_a"].efferent, EXPECTED_SINGLE_COMPONENT)
    expect_true(abs(result["module_a"].instability - INSTABILITY_HALF) < PAGERANK_TOLERANCE)
    expect_true(abs(result["module_b"].instability - INSTABILITY_HALF) < PAGERANK_TOLERANCE)


def test_abstractness_zero_total_returns_zero() -> None:
    """Zero total classes returns zero abstractness."""
    result = compute_abstractness("module", abstract_count=0, total_count=0)
    expect_true(result == 0.0)


def test_abstractness_computes_ratio() -> None:
    """Computes abstract/total ratio."""
    result = compute_abstractness("module", abstract_count=2, total_count=4)
    expect_true(result == INSTABILITY_HALF)


def test_abstractness_all_abstract_returns_one() -> None:
    """All abstract returns 1.0."""
    result = compute_abstractness("module", abstract_count=5, total_count=5)
    expect_true(result == INSTABILITY_FULL)


def test_distance_main_sequence_on_main_returns_zero() -> None:
    """Point on main sequence returns zero distance."""
    coupling = CouplingMetrics(afferent=1, efferent=1, instability=INSTABILITY_HALF)
    result = compute_distance_from_main_sequence(coupling, abstractness=INSTABILITY_HALF)

    expect_true(result == INSTABILITY_ZERO)


def test_distance_main_sequence_off_main_returns_distance() -> None:
    """Point off main sequence returns positive distance."""
    coupling = CouplingMetrics(afferent=0, efferent=1, instability=INSTABILITY_FULL)
    result = compute_distance_from_main_sequence(coupling, abstractness=INSTABILITY_HALF)

    # abs(0.5 + 1.0 - 1.0) = 0.5
    expect_true(result == INSTABILITY_HALF)


@pytest.mark.parametrize(
    ("coupling", "abstractness", "expected"),
    [
        (CouplingMetrics(afferent=5, efferent=0, instability=INSTABILITY_ZERO), 1.0, 0.0),
        (CouplingMetrics(afferent=0, efferent=5, instability=INSTABILITY_FULL), 0.0, 0.0),
        (CouplingMetrics(afferent=5, efferent=0, instability=INSTABILITY_ZERO), 0.0, 1.0),
        (CouplingMetrics(afferent=0, efferent=5, instability=INSTABILITY_FULL), 1.0, 1.0),
        (CouplingMetrics(afferent=3, efferent=3, instability=INSTABILITY_HALF), 0.5, 0.0),
    ],
)
def test_distance_main_sequence_key_scenarios(
    coupling: CouplingMetrics, abstractness: float, expected: float
) -> None:
    """Distance from main sequence handles canonical stability quadrants."""
    result = compute_distance_from_main_sequence(coupling, abstractness=abstractness)

    expect_true(abs(result - expected) < PAGERANK_TOLERANCE)


def test_louvain_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = nx.Graph()
    result = detect_communities_louvain(graph)
    expect_true(result == [])


def test_louvain_disconnected_separate_communities() -> None:
    """Disconnected components are separate communities."""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3)])
    graph.add_edges_from([(10, 11), (11, 12)])
    result = detect_communities_louvain(graph)

    expect_true(len(result) >= EXPECTED_MIN_COMPONENTS)


def test_louvain_returns_community_dataclass() -> None:
    """Returns Community dataclass."""
    graph = complete_graph(5)
    result = detect_communities_louvain(graph)

    expect_true(len(result) >= 1)
    comm = result[0]
    expect_true(isinstance(comm, Community))
    expect_true(hasattr(comm, "community_id"))
    expect_true(hasattr(comm, "nodes"))
    expect_true(hasattr(comm, "size"))


def test_label_propagation_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = nx.Graph()
    result = detect_communities_label_propagation(graph)
    expect_true(result == [])


def test_label_propagation_returns_communities() -> None:
    """Returns communities for connected graph."""
    graph = complete_graph(5)
    result = detect_communities_label_propagation(graph)

    expect_true(len(result) >= 1)
    total_nodes = sum(c.size for c in result)
    expect_true(total_nodes == EXPECTED_NODE_COUNT_FIVE)


def test_modularity_empty_graph_returns_zero() -> None:
    """Empty graph returns zero modularity."""
    graph = nx.Graph()
    result = compute_modularity(graph, [])
    expect_true(result == 0.0)


def test_modularity_empty_communities_returns_zero() -> None:
    """Empty communities returns zero modularity."""
    graph = complete_graph(5)
    result = compute_modularity(graph, [])
    expect_true(result == 0.0)


def test_modularity_in_valid_range() -> None:
    """Modularity is in valid range."""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (10, 11), (11, 12)])
    communities = [
        Community(community_id=0, nodes=frozenset([1, 2, 3]), size=COMMUNITY_SIZE_THREE),
        Community(community_id=1, nodes=frozenset([10, 11, 12]), size=COMMUNITY_SIZE_THREE),
    ]
    result = compute_modularity(graph, communities)

    expect_true(MODULARITY_MIN <= result <= MODULARITY_MAX)


def test_clustering_empty_graph_returns_empty() -> None:
    """Empty graph returns empty dict."""
    graph = nx.Graph()
    result = compute_clustering_coefficient(graph)
    expect_true(result == {})


def test_clustering_complete_graph_full() -> None:
    """Complete graph has full clustering."""
    graph = complete_graph(5)
    result = compute_clustering_coefficient(graph)

    for coeff in result.values():
        expect_true(abs(coeff - 1.0) < PAGERANK_TOLERANCE)


def test_average_clustering() -> None:
    """Average clustering coefficient computation."""
    graph = complete_graph(5)
    result = compute_average_clustering(graph)

    expect_true(abs(result - 1.0) < PAGERANK_TOLERANCE)


def test_average_clustering_empty_graph() -> None:
    """Empty graph returns zero average clustering."""
    graph = nx.Graph()
    result = compute_average_clustering(graph)
    expect_true(result == 0.0)


def test_hub_nodes_empty_graph_returns_empty() -> None:
    """Empty graph returns empty list."""
    graph = nx.Graph()
    result = find_hub_nodes(graph)
    expect_true(result == [])


def test_hub_nodes_star_graph_center_is_hub() -> None:
    """Star graph center is a hub."""
    # Create star with center 0 connected to 10 other nodes
    graph = nx.star_graph(STAR_GRAPH_SIZE_TEN)
    result = find_hub_nodes(graph, min_degree=MIN_HUB_DEGREE)

    expect_true(0 in result)  # Center node


def test_hub_nodes_threshold_ratio_parameter() -> None:
    """Threshold ratio parameter works."""
    graph = nx.star_graph(STAR_GRAPH_SIZE_TEN)
    result_strict = find_hub_nodes(graph, threshold_ratio=0.9, min_degree=1)
    result_loose = find_hub_nodes(graph, threshold_ratio=HUB_THRESHOLD_RATIO, min_degree=1)

    # Stricter threshold should find fewer hubs
    expect_true(len(result_strict) <= len(result_loose))


def test_boundary_nodes_empty_communities_returns_empty() -> None:
    """Empty communities returns empty list."""
    graph = nx.Graph()
    result = find_boundary_nodes(graph, [])
    expect_true(result == [])


def test_boundary_nodes_finds_boundary() -> None:
    """Finds nodes at community boundaries."""
    graph = nx.Graph()
    # Two communities with bridge node
    graph.add_edges_from([(1, 2), (2, 3), (3, 10), (10, 11), (11, 12)])
    communities = [
        Community(community_id=0, nodes=frozenset([1, 2, 3]), size=COMMUNITY_SIZE_THREE),
        Community(community_id=1, nodes=frozenset([10, 11, 12]), size=COMMUNITY_SIZE_THREE),
    ]
    result = find_boundary_nodes(graph, communities)

    # Nodes 3 and 10 are at the boundary
    expect_true(EXPECTED_CYCLE_NODES in result or STAR_GRAPH_SIZE_TEN in result)


def test_coupling_to_rows_converts_metrics() -> None:
    """Converts CouplingMetrics to row dicts."""
    metrics = {
        "module_a": CouplingMetrics(
            afferent=AFFERENT_THREE, efferent=EFFERENT_TWO, instability=INSTABILITY_POINT_FOUR
        ),
    }
    rows = coupling_to_rows(metrics, repo="test-repo", commit="abc123")

    expect_true(len(rows) == 1)
    row = rows[0]
    expect_true(row["module"] == "module_a")
    expect_true(row["repo"] == "test-repo")
    expect_true(row["commit"] == "abc123")
    expect_true(row["afferent_coupling"] == AFFERENT_THREE)
    expect_true(row["efferent_coupling"] == EFFERENT_TWO)
    expect_true(row["instability"] == INSTABILITY_POINT_FOUR)


# ===========================================================================
# STRUCTURAL METRIC TESTS
# ===========================================================================


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
    graph = nx.Graph()
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
    graph = nx.Graph()
    result = compute_all_structural(graph)

    expect_true(result == {})


# ===========================================================================
# DATACLASS TESTS
# ===========================================================================


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


def test_coupling_metrics_frozen() -> None:
    """CouplingMetrics is frozen."""
    metrics = CouplingMetrics(
        afferent=AFFERENT_THREE, efferent=EFFERENT_TWO, instability=INSTABILITY_POINT_FOUR
    )
    assert_cannot_setattr(metrics, "afferent", 5)


def test_community_frozen() -> None:
    """Community is frozen."""
    comm = Community(community_id=0, nodes=frozenset([1, 2, 3]), size=COMMUNITY_SIZE_THREE)
    assert_cannot_setattr(comm, "size", 5)


def test_scc_result_frozen() -> None:
    """SCCResult is frozen."""
    result = SCCResult(components=(), node_to_component={})
    assert_cannot_setattr(result, "condensation", nx.DiGraph())


# ===========================================================================
# REALISTIC GOLDEN DATASET TESTS
# ===========================================================================
# These tests use production-realistic graph structures from the golden dataset
# to ensure algorithms work correctly on complex, realistic data.


GOLDEN_MIN_NODES: Final[int] = 13
GOLDEN_MIN_EDGES: Final[int] = 30
GOLDEN_EXPECTED_COMMUNITIES: Final[int] = 2
GOLDEN_EXPECTED_SCC: Final[int] = 1


def _build_realistic_call_graph() -> nx.DiGraph:
    """Build a realistic call graph simulating production patterns.

    Returns
    -------
    nx.DiGraph
        A directed graph with hub functions, layered architecture, and SCCs.
    """
    g = nx.DiGraph()

    # Layer 0: Core utilities (no internal deps)
    core_funcs = ["format_string", "parse_json", "validate_input", "hash_value"]
    g.add_nodes_from(core_funcs)

    # Layer 1: Services (depend on core)
    services = ["authenticate", "query", "execute", "get_cached", "set_cached"]
    g.add_nodes_from(services)
    for s in services:
        g.add_edge(s, "validate_input")
        g.add_edge(s, "format_string")

    # Layer 2: Handlers (depend on services, core)
    handlers = ["create_user", "get_user", "update_user", "delete_user", "create_order"]
    g.add_nodes_from(handlers)
    for h in handlers:
        g.add_edge(h, "authenticate")
        g.add_edge(h, "query")
        g.add_edge(h, "get_cached")

    # Layer 3: API (depend on handlers)
    api = ["handle_request", "register_routes"]
    g.add_nodes_from(api)
    for a in api:
        for h in handlers:
            g.add_edge(a, h)

    # Hub function: log_info is called by many
    g.add_node("log_info")
    for node in services + handlers:
        g.add_edge(node, "log_info")

    # Small SCC: auth <-> cache interaction
    g.add_edge("authenticate", "get_cached")
    g.add_edge("get_cached", "authenticate")  # Cache validates with auth

    return g


def _build_realistic_import_graph() -> nx.DiGraph:
    """Build a realistic import graph with layered architecture.

    Returns
    -------
    nx.DiGraph
        A directed graph representing module imports.
    """
    g = nx.DiGraph()

    # Core modules
    core = ["core.utils", "core.types", "core.errors", "core.config"]
    g.add_nodes_from(core)

    # Service modules
    services = ["services.auth", "services.cache", "services.database"]
    g.add_nodes_from(services)
    for s in services:
        g.add_edge(s, "core.utils")
        g.add_edge(s, "core.errors")

    # Handler modules
    handlers = ["handlers.user", "handlers.product", "handlers.order"]
    g.add_nodes_from(handlers)
    for h in handlers:
        g.add_edge(h, "services.auth")
        g.add_edge(h, "services.database")
        g.add_edge(h, "core.errors")

    # API modules
    api = ["api.routes", "api.middleware"]
    g.add_nodes_from(api)
    for a in api:
        for h in handlers:
            g.add_edge(a, h)

    # Cross-cutting: utils.logging imported by many
    g.add_node("utils.logging")
    for node in services + handlers + api:
        g.add_edge(node, "utils.logging")

    # Intentional cycle: services.auth <-> services.cache
    g.add_edge("services.auth", "services.cache")
    g.add_edge("services.cache", "services.auth")

    return g


def test_realistic_pagerank_identifies_hub_functions() -> None:
    """PageRank correctly identifies hub functions in realistic graphs."""
    graph = _build_realistic_call_graph()

    result = compute_pagerank(graph)

    # Hub functions should have higher PageRank
    # log_info is called by many, so should have high rank
    hub_rank = result.get("log_info", 0)

    # Hub should have meaningful PageRank
    expect_true(hub_rank > 0)
    # Ensure we got results for multiple nodes
    expect_true(len(result) >= GOLDEN_MIN_NODES)


def test_realistic_scc_finds_cycles() -> None:
    """SCC detection correctly identifies cycles in realistic graphs."""
    graph = _build_realistic_call_graph()

    result = find_strongly_connected(graph)

    # Should have at least one SCC (the auth-cache cycle)
    # Most SCCs will be single nodes (trivial), but at least one should have >1 node
    non_trivial_sccs = [comp for comp in result.components if comp.size > 1]
    expect_true(len(non_trivial_sccs) >= GOLDEN_EXPECTED_SCC)


def test_realistic_import_layers_computed() -> None:
    """Topological layers work on realistic import graphs."""
    graph = _build_realistic_import_graph()

    # Need DAG for topological layers, so we use condensation first
    scc_result = find_strongly_connected(graph, compute_condensation=True)

    # The graph has a cycle, so use condensation layers
    layers = condensation_layers(graph, scc_result)

    # Should have multiple layers due to the layered architecture
    expect_true(len(layers) >= EXPECTED_NODE_COUNT_TWO)


def test_realistic_centrality_metrics() -> None:
    """All centrality metrics work on realistic graphs."""
    graph = _build_realistic_call_graph()

    metrics = compute_all_centralities(graph)

    # Should have metrics for all nodes
    expect_true(len(metrics) >= GOLDEN_MIN_NODES)

    # All metric values should be in valid ranges
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

    # Should have valid statistics
    expect_true(stats["count"] >= GOLDEN_EXPECTED_SCC)
    expect_true(stats["mean_size"] > 0)
    expect_true(stats["largest_size"] >= 1)


def test_realistic_community_detection() -> None:
    """Community detection finds meaningful communities in realistic graphs."""
    graph = _build_realistic_import_graph().to_undirected()

    communities = detect_communities_louvain(graph)

    # Should find at least 2 communities (core/services vs handlers/api)
    expect_true(len(communities) >= GOLDEN_EXPECTED_COMMUNITIES)

    # All nodes should be assigned to a community
    all_nodes = set(graph.nodes())
    community_nodes = set()
    for comm in communities:
        community_nodes.update(comm.nodes)
    expect_true(all_nodes == community_nodes)


def test_realistic_hub_detection() -> None:
    """Hub detection finds high-connectivity nodes in realistic graphs."""
    graph = _build_realistic_import_graph()

    hubs = find_hub_nodes(graph, min_degree=3, threshold_ratio=0.05)

    # Should find some hubs (modules imported by many)
    expect_true(len(hubs) >= 1)

    # Check that we have actual hub node names
    expect_true(all(isinstance(h, str) for h in hubs))


def test_realistic_coupling_metrics() -> None:
    """Coupling metrics work on realistic import graphs."""
    graph = _build_realistic_import_graph()

    # Test coupling for all nodes
    all_metrics = compute_coupling(graph)

    # Should have metrics for all nodes
    expect_true(len(all_metrics) >= GOLDEN_MIN_NODES)

    # Test specific module
    metrics = all_metrics.get("handlers.user")
    expect_is_not_none(metrics)
    if metrics is None:
        return

    # Handler modules have both afferent (api imports them) and efferent (import services)
    expect_true(metrics.afferent >= 1)  # At least api imports it
    expect_true(metrics.efferent >= 1)  # At least imports services.auth
    expect_true(metrics.instability >= 0)
    expect_true(metrics.instability <= 1)
