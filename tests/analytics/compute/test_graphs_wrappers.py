"""Analytics graph wrapper computations."""

from __future__ import annotations

from datetime import UTC, datetime

import networkx as nx

from codeintel.analytics.compute.graphs.centrality import (
    centrality_directed,
    centrality_undirected,
    neighbor_stats,
)
from codeintel.analytics.compute.graphs.cfg import (
    build_cfg_graph,
    cfg_avg_shortest_path_length,
    cfg_centralities,
    cfg_dominance_metrics,
    cfg_longest_path_length,
    cfg_reachable_nodes,
)
from codeintel.analytics.compute.graphs.components import (
    component_ids_undirected,
    component_metadata,
    global_graph_stats,
)
from codeintel.analytics.compute.graphs.dfg import (
    build_dfg_graph,
    dfg_centralities,
    dfg_component_stats,
    dfg_path_lengths,
)
from codeintel.analytics.compute.graphs.projections import (
    bipartite_degrees,
    build_projection_graph,
    projection_metrics,
)
from codeintel.analytics.compute.graphs.structural import (
    bounded_simple_path_count,
    structural_metrics,
)
from codeintel.analytics.runtime.context import GraphContext
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_true,
)


def _context() -> GraphContext:
    return GraphContext(repo="demo", commit="abc", now=datetime.now(tz=UTC), seed=7)


def test_centrality_wrappers_and_neighbor_stats() -> None:
    """Compute centrality bundles and neighbor stats on small graphs."""
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [
            ("a", "b", 2),
            ("b", "c", 1),
            ("c", "a", 3),
        ]
    )
    stats = neighbor_stats(graph, weight="weight")
    expect_equal(stats.out_counts["a"], 2)
    expect_equal(stats.in_counts["a"], 3)

    ctx = _context()
    directed = centrality_directed(graph, ctx, include_eigen=True)
    expect_equal(set(directed.pagerank), {"a", "b", "c"})
    expect_true(directed.eigenvector)

    undirected_graph = nx.Graph()
    undirected_graph.add_edge("x", "y", weight=1)
    undirected_graph.add_edge("y", "z", weight=2)
    undirected = centrality_undirected(
        undirected_graph,
        ctx,
        include_structural=True,
    )
    expect_true(undirected.closeness)
    expect_true(undirected.eigenvector)


def test_structural_metrics_and_paths() -> None:
    """Evaluate structural metrics for empty and populated graphs."""
    empty = structural_metrics(nx.Graph())
    expect_equal(empty.clustering, {})
    expect_equal(empty.community_id, {})

    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
    populated = structural_metrics(graph, community_limit=10)
    expect_equal(set(populated.clustering), {1, 2, 3, 4})
    expect_true(populated.community_id)

    digraph = nx.DiGraph()
    digraph.add_edges_from([(1, 2), (2, 3)])
    bounded = bounded_simple_path_count(digraph, sources=[1], targets=[3], max_paths=5, cutoff=3)
    expect_true(bounded > 0)


def test_component_metadata_and_global_stats() -> None:
    """Compute component metadata and global statistics."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 1), (3, 4)])
    metadata = component_metadata(graph)
    expect_true(metadata.in_cycle[1])
    expect_equal(metadata.component_size[3], 2)

    undirected = nx.Graph()
    undirected.add_edges_from([(10, 11), (12, 13)])
    comp_id, comp_size = component_ids_undirected(undirected)
    expect_equal(comp_id[10], comp_id[11])
    expect_equal(comp_size[12], 2)

    stats = global_graph_stats(undirected)
    expect_equal(stats.node_count, 4)
    expect_equal(stats.weak_component_count, 2)


def test_cfg_helpers_and_metrics() -> None:
    """Validate CFG helpers including dominance and centrality."""
    blocks = [
        (0, "entry", 0, 1),
        (1, "body", 1, 1),
        (2, "exit", 1, 0),
    ]
    edges = [(0, 1, "next"), (1, 2, "next")]
    graph, entry_idx, exit_idx = build_cfg_graph(blocks, edges)
    expect_equal(entry_idx, 0)
    expect_equal(exit_idx, 2)
    dominance = cfg_dominance_metrics(graph, entry_idx)
    expect_equal(dominance.tree_height, 2)

    ctx = _context()
    centrality, dom = cfg_centralities(graph, entry_idx, ctx=ctx)
    expect_true(dom.depth)
    expect_true(centrality.pagerank)

    longest = cfg_longest_path_length(graph, entry_idx, is_dag=True)
    expect_equal(longest, 2)
    avg = cfg_avg_shortest_path_length(graph, entry_idx)
    expect_true(avg > 0)
    reachable = cfg_reachable_nodes(graph, entry_idx)
    expect_equal(reachable, {0, 1, 2})


def test_dfg_helpers_and_metrics() -> None:
    """Validate DFG helpers including path lengths and centrality."""
    edges = [
        (1, 2, "a", "b", True, "use"),
        (2, 3, "b", "c", False, "use"),
    ]
    graph, phi_edges, symbol_count = build_dfg_graph(edges)
    expect_equal(phi_edges, 1)
    expect_equal(symbol_count, 3)

    component_count, components, has_cycles = dfg_component_stats(graph)
    expect_equal(component_count, 1)
    expect_equal(components[0], {1, 2, 3})
    expect_false(has_cycles)

    longest, average = dfg_path_lengths(graph)
    expect_equal(longest, 2)
    expect_true(average > 0.0)

    ctx = _context()
    betweenness, eigen = dfg_centralities(graph, ctx)
    expect_true(betweenness)
    expect_true(eigen)


def test_projection_metrics_and_degrees() -> None:
    """Compute projection metrics and bipartite degrees."""
    bipartite = nx.Graph()
    bipartite.add_edges_from(
        [
            ("user1", "repo1"),
            ("user1", "repo2"),
            ("user2", "repo2"),
        ]
    )
    projection = build_projection_graph(bipartite, {"user1", "user2"}, label="users")
    expect_equal(projection.number_of_nodes(), 2)

    ctx = _context()
    metrics = projection_metrics(
        bipartite,
        {"user1", "user2"},
        ctx,
        projection=projection,
        label="users",
    )
    expect_true(metrics.degree)
    expect_true(metrics.weighted_degree)
    expect_true(metrics.community_id)

    degrees = bipartite_degrees(bipartite, {"user1", "user2"}, {"repo1", "repo2"})
    expect_equal(degrees.degree["user1"], 2)
    expect_true(degrees.secondary_degree_centrality["repo2"] > 0)
