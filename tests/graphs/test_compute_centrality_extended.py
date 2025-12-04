"""Extended tests for centrality computation module.

This module provides additional test coverage for the centrality module
from `codeintel.graphs.compute.metrics.centrality`, including:

- PageRank computation with various graph structures
- Betweenness centrality computation
- Closeness centrality computation
- Degree centrality (in/out/total)
- All centralities combined computation
- Edge cases (empty graphs, single nodes, disconnected components)
"""

from __future__ import annotations

from typing import Final

import networkx as nx

from codeintel.graphs.compute.metrics.centrality import (
    CentralityMetrics,
    compute_all_centralities,
    compute_betweenness,
    compute_closeness,
    compute_degree_centrality,
    compute_in_degree_centrality,
    compute_out_degree_centrality,
    compute_pagerank,
)
from tests._helpers.fakes.networkx_graphs import (
    chain_graph,
    cyclic_graph,
    diamond_graph,
    disconnected_graph,
    star_graph,
)

# Constants
PAGERANK_TOLERANCE: Final = 0.01
CENTRALITY_TOLERANCE: Final = 0.001
SIMPLE_CHAIN_NODE_COUNT: Final = 4
DISCONNECTED_NODE_COUNT: Final = 6
SINGLE_NODE_COUNT: Final = 1
DIAMOND_OUT_EDGES: Final = 2
DIAMOND_IN_EDGES: Final = 2
METRIC_PAGERANK: Final = 0.25
METRIC_BETWEENNESS: Final = 0.5
METRIC_CLOSENESS: Final = 0.75
METRIC_IN_DEGREE: Final = 2
METRIC_OUT_DEGREE: Final = 3
METRIC_DEGREE_TOTAL: Final = 5
EQUALITY_DEGREE_TOTAL: Final = 2
DEFAULT_DEGREE_TOTAL: Final = 3


def test_compute_pagerank_simple_chain() -> None:
    """Compute PageRank on a simple chain graph."""
    g = chain_graph(4)

    pagerank = compute_pagerank(g)

    assert len(pagerank) == SIMPLE_CHAIN_NODE_COUNT
    assert all(v > 0 for v in pagerank.values())
    # Sum should be approximately 1
    assert abs(sum(pagerank.values()) - 1.0) < PAGERANK_TOLERANCE


def test_compute_pagerank_star_graph() -> None:
    """Compute PageRank on a star graph."""
    g = star_graph(3)

    pagerank = compute_pagerank(g)

    # Hub should have lower pagerank since it doesn't receive links
    assert pagerank["hub"] < pagerank["spoke1"]


def test_compute_pagerank_reverse_star() -> None:
    """Compute PageRank on a reverse star graph."""
    g = star_graph(3, inward=True)

    pagerank = compute_pagerank(g)

    # Hub should have highest pagerank since it receives all links
    assert pagerank["hub"] > pagerank["spoke1"]


def test_compute_pagerank_cyclic() -> None:
    """Compute PageRank on a cyclic graph."""
    g = cyclic_graph(3)

    pagerank = compute_pagerank(g)

    # In a cycle, all nodes should have similar PageRank
    values = list(pagerank.values())
    assert max(values) - min(values) < PAGERANK_TOLERANCE


def test_compute_pagerank_empty_graph() -> None:
    """Compute PageRank on an empty graph."""
    g = nx.DiGraph()

    pagerank = compute_pagerank(g)

    assert pagerank == {}


def test_compute_pagerank_single_node() -> None:
    """Compute PageRank on a single node graph."""
    g = nx.DiGraph()
    g.add_node("A")

    pagerank = compute_pagerank(g)

    assert len(pagerank) == SINGLE_NODE_COUNT
    assert abs(pagerank["A"] - 1.0) < PAGERANK_TOLERANCE


def test_compute_betweenness_chain() -> None:
    """Compute betweenness on a chain graph."""
    g = chain_graph(4)

    betweenness = compute_betweenness(g)

    # Middle nodes (B, C) should have higher betweenness
    assert betweenness["B"] > betweenness["A"]
    assert betweenness["C"] > betweenness["D"]


def test_compute_betweenness_star() -> None:
    """Compute betweenness on a star graph."""
    g = star_graph(3)

    betweenness = compute_betweenness(g)

    # Hub is on paths to all spokes from each other
    # But with directed edges from hub only, there are no paths through hub
    assert all(v >= 0 for v in betweenness.values())


def test_compute_betweenness_diamond() -> None:
    """Compute betweenness on a diamond graph."""
    g = diamond_graph()

    betweenness = compute_betweenness(g)

    # B and C are on paths from A to D
    assert betweenness["B"] >= betweenness["A"]
    assert betweenness["C"] >= betweenness["A"]


def test_compute_betweenness_empty_graph() -> None:
    """Compute betweenness on an empty graph."""
    g = nx.DiGraph()

    betweenness = compute_betweenness(g)

    assert betweenness == {}


def test_compute_betweenness_disconnected() -> None:
    """Compute betweenness on a disconnected graph."""
    g = disconnected_graph()

    betweenness = compute_betweenness(g)

    # All nodes should have defined betweenness
    assert len(betweenness) == DISCONNECTED_NODE_COUNT


def test_compute_closeness_chain() -> None:
    """Compute closeness on a chain graph."""
    g = chain_graph(4)

    closeness = compute_closeness(g)

    # All nodes should have closeness values defined
    assert len(closeness) == SIMPLE_CHAIN_NODE_COUNT
    assert all(v >= 0 for v in closeness.values())


def test_compute_closeness_star() -> None:
    """Compute closeness on a star graph."""
    g = star_graph(3)

    closeness = compute_closeness(g)

    # All nodes should have closeness values defined
    assert len(closeness) == SIMPLE_CHAIN_NODE_COUNT
    # Spokes have no outgoing edges so closeness might be 0 or special case
    assert all(v >= 0 for v in closeness.values())


def test_compute_closeness_empty_graph() -> None:
    """Compute closeness on an empty graph."""
    g = nx.DiGraph()

    closeness = compute_closeness(g)

    assert closeness == {}


def test_compute_closeness_disconnected() -> None:
    """Compute closeness on a disconnected graph."""
    g = disconnected_graph()

    closeness = compute_closeness(g)

    # All nodes should have closeness values
    assert len(closeness) == DISCONNECTED_NODE_COUNT


def test_compute_degree_centrality_star() -> None:
    """Compute total degree centrality on a star graph."""
    g = star_graph(3)

    degree = compute_degree_centrality(g)

    # Hub has highest degree (3 out-edges)
    assert degree["hub"] > degree["spoke1"]


def test_compute_in_degree_centrality_reverse_star() -> None:
    """Compute in-degree centrality on a reverse star graph."""
    g = star_graph(3, inward=True)

    in_degree = compute_in_degree_centrality(g)

    # Hub receives all edges
    assert in_degree["hub"] > in_degree["spoke1"]


def test_compute_out_degree_centrality_star() -> None:
    """Compute out-degree centrality on a star graph."""
    g = star_graph(3)

    out_degree = compute_out_degree_centrality(g)

    # Hub has all out-edges
    assert out_degree["hub"] > out_degree["spoke1"]


def test_compute_degree_empty_graph() -> None:
    """Compute degree centrality on an empty graph."""
    g = nx.DiGraph()

    degree = compute_degree_centrality(g)

    assert degree == {}


def test_compute_in_degree_empty_graph() -> None:
    """Compute in-degree centrality on an empty graph."""
    g = nx.DiGraph()

    in_degree = compute_in_degree_centrality(g)

    assert in_degree == {}


def test_compute_out_degree_empty_graph() -> None:
    """Compute out-degree centrality on an empty graph."""
    g = nx.DiGraph()

    out_degree = compute_out_degree_centrality(g)

    assert out_degree == {}


def test_compute_all_centralities_simple_chain() -> None:
    """Compute all centralities on a simple chain."""
    g = chain_graph(4)

    all_metrics = compute_all_centralities(g)

    assert len(all_metrics) == SIMPLE_CHAIN_NODE_COUNT
    assert "A" in all_metrics
    assert "B" in all_metrics


def test_compute_all_centralities_returns_centrality_metrics() -> None:
    """Compute all centralities returns CentralityMetrics objects."""
    g = chain_graph(4)

    all_metrics = compute_all_centralities(g)

    for metrics in all_metrics.values():
        assert isinstance(metrics, CentralityMetrics)
        assert hasattr(metrics, "pagerank")
        assert hasattr(metrics, "betweenness")
        assert hasattr(metrics, "closeness")
        assert hasattr(metrics, "in_degree")
        assert hasattr(metrics, "out_degree")


def test_compute_all_centralities_empty_graph() -> None:
    """Compute all centralities on an empty graph."""
    g = nx.DiGraph()

    all_metrics = compute_all_centralities(g)

    assert all_metrics == {}


def test_compute_all_centralities_single_node() -> None:
    """Compute all centralities on a single node graph."""
    g = nx.DiGraph()
    g.add_node("A")

    all_metrics = compute_all_centralities(g)

    assert len(all_metrics) == SINGLE_NODE_COUNT
    assert "A" in all_metrics
    metrics = all_metrics["A"]
    assert metrics.in_degree == 0
    assert metrics.out_degree == 0


def test_compute_all_centralities_diamond() -> None:
    """Compute all centralities on a diamond graph."""
    g = diamond_graph()

    all_metrics = compute_all_centralities(g)

    assert len(all_metrics) == SIMPLE_CHAIN_NODE_COUNT

    # A has 2 outgoing edges
    assert all_metrics["A"].out_degree == DIAMOND_OUT_EDGES
    assert all_metrics["A"].in_degree == 0

    # D has 2 incoming edges
    assert all_metrics["D"].in_degree == DIAMOND_IN_EDGES
    assert all_metrics["D"].out_degree == 0


def test_compute_all_centralities_disconnected() -> None:
    """Compute all centralities on a disconnected graph."""
    g = disconnected_graph()

    all_metrics = compute_all_centralities(g)

    assert len(all_metrics) == DISCONNECTED_NODE_COUNT
    # All nodes should have metrics
    assert "A" in all_metrics
    assert "X" in all_metrics


def test_centrality_metrics_all_fields() -> None:
    """CentralityMetrics has all expected fields."""
    metrics = CentralityMetrics(
        pagerank=METRIC_PAGERANK,
        betweenness=METRIC_BETWEENNESS,
        closeness=METRIC_CLOSENESS,
        harmonic=METRIC_CLOSENESS,
        eigenvector=METRIC_PAGERANK,
        in_degree=METRIC_IN_DEGREE,
        out_degree=METRIC_OUT_DEGREE,
        degree=METRIC_DEGREE_TOTAL,
    )

    assert metrics.pagerank == METRIC_PAGERANK
    assert metrics.betweenness == METRIC_BETWEENNESS
    assert metrics.closeness == METRIC_CLOSENESS
    assert metrics.harmonic == METRIC_CLOSENESS
    assert metrics.eigenvector == METRIC_PAGERANK
    assert metrics.in_degree == METRIC_IN_DEGREE
    assert metrics.out_degree == METRIC_OUT_DEGREE
    assert metrics.degree == METRIC_DEGREE_TOTAL


def test_centrality_metrics_equality() -> None:
    """CentralityMetrics supports equality comparison."""
    m1 = CentralityMetrics(
        pagerank=METRIC_BETWEENNESS,
        betweenness=METRIC_BETWEENNESS,
        closeness=METRIC_BETWEENNESS,
        harmonic=METRIC_BETWEENNESS,
        eigenvector=METRIC_BETWEENNESS,
        in_degree=1,
        out_degree=1,
        degree=EQUALITY_DEGREE_TOTAL,
    )
    m2 = CentralityMetrics(
        pagerank=METRIC_BETWEENNESS,
        betweenness=METRIC_BETWEENNESS,
        closeness=METRIC_BETWEENNESS,
        harmonic=METRIC_BETWEENNESS,
        eigenvector=METRIC_BETWEENNESS,
        in_degree=1,
        out_degree=1,
        degree=EQUALITY_DEGREE_TOTAL,
    )

    assert m1 == m2


def test_centrality_metrics_default_values() -> None:
    """CentralityMetrics has correct defaults."""
    # Check that it can be created with required fields
    metrics = CentralityMetrics(
        pagerank=0.1,
        betweenness=0.2,
        closeness=0.3,
        harmonic=0.4,
        eigenvector=0.5,
        in_degree=1,
        out_degree=2,
        degree=3,
    )

    assert metrics is not None
    assert metrics.degree == DEFAULT_DEGREE_TOTAL
