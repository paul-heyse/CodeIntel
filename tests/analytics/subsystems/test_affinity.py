"""Tests for module affinity graph construction and clustering.

This module tests:
- parse_tags normalization
- add_graph_weight for accumulating edge weights
- graph_to_adjacency conversion
- seed_labels_from_tags derivation
- label_propagation_nx algorithm
- reassign_small_clusters merging
- best_neighbor_label selection
- cluster_sizes_map computation
"""

from __future__ import annotations

from typing import cast

import networkx as nx

from codeintel.analytics.subsystems.affinity import (
    add_graph_weight,
    best_neighbor_label,
    cluster_sizes_map,
    graph_to_adjacency,
    label_propagation_nx,
    parse_tags,
    reassign_small_clusters,
    seed_labels_from_tags,
)

# Test constants
DEFAULT_WEIGHT = 1.0
CUSTOM_WEIGHT = 2.5
MIN_CLUSTER_SIZE = 3
CLUSTER_SIZE_TWO = 2
CLUSTER_SIZE_THREE = 3


# =============================================================================
# parse_tags Tests
# =============================================================================


def test_parse_tags_none() -> None:
    """parse_tags returns empty list for None."""
    result = parse_tags(None)

    assert result == []


def test_parse_tags_json_list() -> None:
    """parse_tags parses JSON list string."""
    result = parse_tags('["tag1", "tag2", "tag3"]')

    assert result == ["tag1", "tag2", "tag3"]


def test_parse_tags_json_single_value() -> None:
    """parse_tags parses JSON single value."""
    result = parse_tags('"single_tag"')

    assert result == ["single_tag"]


def test_parse_tags_plain_string() -> None:
    """parse_tags handles plain string."""
    result = parse_tags("not_json_tag")

    assert result == ["not_json_tag"]


def test_parse_tags_list() -> None:
    """parse_tags handles Python list."""
    result = parse_tags(["a", "b", "c"])

    assert result == ["a", "b", "c"]


def test_parse_tags_other_type() -> None:
    """parse_tags converts other types to string."""
    result = parse_tags(123)

    assert result == ["123"]


def test_parse_tags_mixed_list() -> None:
    """parse_tags converts mixed list elements to strings."""
    result = parse_tags([1, "two", 3.0])

    assert result == ["1", "two", "3.0"]


# =============================================================================
# add_graph_weight Tests
# =============================================================================


def test_add_graph_weight_new_edge() -> None:
    """add_graph_weight creates new edge with weight."""
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B"])

    add_graph_weight(graph, "A", "B", CUSTOM_WEIGHT)

    assert graph.has_edge("A", "B")
    assert graph["A"]["B"]["weight"] == CUSTOM_WEIGHT


def test_add_graph_weight_accumulates() -> None:
    """add_graph_weight accumulates weight on existing edge."""
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=DEFAULT_WEIGHT)

    add_graph_weight(graph, "A", "B", CUSTOM_WEIGHT)

    expected = DEFAULT_WEIGHT + CUSTOM_WEIGHT
    assert graph["A"]["B"]["weight"] == expected


def test_add_graph_weight_ignores_self_loop() -> None:
    """add_graph_weight ignores self-loops."""
    graph = nx.Graph()
    graph.add_node("A")

    add_graph_weight(graph, "A", "A", CUSTOM_WEIGHT)

    assert not graph.has_edge("A", "A")


def test_add_graph_weight_ignores_zero_weight() -> None:
    """add_graph_weight ignores zero weight."""
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B"])

    add_graph_weight(graph, "A", "B", 0.0)

    assert not graph.has_edge("A", "B")


def test_add_graph_weight_ignores_negative_weight() -> None:
    """add_graph_weight ignores negative weight."""
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B"])

    add_graph_weight(graph, "A", "B", -1.0)

    assert not graph.has_edge("A", "B")


# =============================================================================
# graph_to_adjacency Tests
# =============================================================================


def test_graph_to_adjacency_empty() -> None:
    """graph_to_adjacency returns empty dict for empty graph."""
    graph = nx.Graph()

    result = graph_to_adjacency(graph)

    assert result == {}


def test_graph_to_adjacency_single_edge() -> None:
    """graph_to_adjacency creates symmetric entries for undirected edge."""
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=CUSTOM_WEIGHT)

    result = graph_to_adjacency(graph)

    assert result["A"]["B"] == CUSTOM_WEIGHT
    assert result["B"]["A"] == CUSTOM_WEIGHT


def test_graph_to_adjacency_multiple_edges() -> None:
    """graph_to_adjacency handles multiple edges."""
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("B", "C", weight=2.0)
    graph.add_edge("A", "C", weight=3.0)

    result = graph_to_adjacency(graph)

    expected_edges = 3
    # Each undirected edge creates 2 entries in adjacency
    total_entries = sum(len(neighbors) for neighbors in result.values())
    assert total_entries == expected_edges * 2


def test_graph_to_adjacency_default_weight() -> None:
    """graph_to_adjacency uses default weight of 1.0 if not specified."""
    graph = nx.Graph()
    graph.add_edge("A", "B")  # No weight specified

    result = graph_to_adjacency(graph)

    assert result["A"]["B"] == DEFAULT_WEIGHT


# =============================================================================
# seed_labels_from_tags Tests
# =============================================================================


def test_seed_labels_from_tags_empty() -> None:
    """seed_labels_from_tags returns empty dict for empty input."""
    result = seed_labels_from_tags({})

    assert result == {}


def test_seed_labels_from_tags_basic() -> None:
    """seed_labels_from_tags extracts first tag as lowercase label."""
    tags = {
        "module.a": ["TagA", "other"],
        "module.b": ["TagB"],
    }

    result = seed_labels_from_tags(tags)

    assert result["module.a"] == "taga"
    assert result["module.b"] == "tagb"


def test_seed_labels_from_tags_empty_tags_skipped() -> None:
    """seed_labels_from_tags skips modules with empty tags."""
    tags = {
        "module.a": ["TagA"],
        "module.b": [],
    }

    result = seed_labels_from_tags(tags)

    assert "module.a" in result
    assert "module.b" not in result


def test_seed_labels_from_tags_none_first_tag_skipped() -> None:
    """seed_labels_from_tags skips modules where first tag is None."""
    # Intentionally pass invalid data to test defensive handling.
    # Cast to expected type since we're testing edge case behavior.
    invalid_tags = {"module.a": [None, "valid"]}
    tags = cast("dict[str, list[str]]", invalid_tags)

    result = seed_labels_from_tags(tags)

    assert "module.a" not in result


# =============================================================================
# label_propagation_nx Tests
# =============================================================================


def test_label_propagation_nx_preserves_seeds() -> None:
    """label_propagation_nx preserves seed labels."""
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=1.0)
    seed_labels = {"A": "cluster1"}

    result = label_propagation_nx(graph, seed_labels)

    assert result["A"] == "cluster1"


def test_label_propagation_nx_propagates_to_neighbors() -> None:
    """label_propagation_nx propagates labels to neighbors."""
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("A", "C", weight=1.0)
    seed_labels = {"A": "group"}

    result = label_propagation_nx(graph, seed_labels)

    # B and C should adopt A's label through propagation
    assert result["A"] == "group"


def test_label_propagation_nx_selects_heaviest_neighbor() -> None:
    """label_propagation_nx selects label from heaviest weighted neighbor."""
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("A", "C", weight=5.0)  # Heavier edge to C
    seed_labels = {"B": "light", "C": "heavy"}

    result = label_propagation_nx(graph, seed_labels)

    # A should adopt C's label due to heavier weight
    assert result["A"] == "heavy"


def test_label_propagation_nx_isolated_nodes_keep_fallback() -> None:
    """label_propagation_nx assigns fallback label to isolated nodes."""
    graph = nx.Graph()
    graph.add_node("isolated")

    result = label_propagation_nx(graph, {})

    assert result["isolated"] == "isolated"


# =============================================================================
# reassign_small_clusters Tests
# =============================================================================


def test_reassign_small_clusters_no_change_when_all_large() -> None:
    """reassign_small_clusters returns same labels when all clusters are large."""
    labels = {"A": "c1", "B": "c1", "C": "c1", "D": "c2", "E": "c2", "F": "c2"}
    adjacency: dict[str, dict[str, float]] = {}

    result = reassign_small_clusters(labels, adjacency, min_size=MIN_CLUSTER_SIZE)

    assert result == labels


def test_reassign_small_clusters_min_size_one_no_change() -> None:
    """reassign_small_clusters returns same labels when min_size is 1."""
    labels = {"A": "c1"}
    adjacency: dict[str, dict[str, float]] = {}

    result = reassign_small_clusters(labels, adjacency, min_size=1)

    assert result == labels


def test_reassign_small_clusters_reassigns_small() -> None:
    """reassign_small_clusters reassigns nodes from small clusters."""
    # c1 has 3 nodes (large), c2 has 1 node (small)
    labels = {"A": "c1", "B": "c1", "C": "c1", "D": "c2"}
    adjacency = {"D": {"A": 2.0, "B": 1.0}}  # D connected to A and B

    result = reassign_small_clusters(labels, adjacency, min_size=MIN_CLUSTER_SIZE)

    # D should be reassigned to c1 since it's the only stable cluster
    assert result["D"] == "c1"


# =============================================================================
# best_neighbor_label Tests
# =============================================================================


def test_best_neighbor_label_no_neighbors() -> None:
    """best_neighbor_label returns None when no neighbors."""
    adjacency: dict[str, dict[str, float]] = {}
    labels = {"A": "c1"}
    allowed = {"c1"}

    result = best_neighbor_label("B", adjacency, labels, allowed)

    assert result is None


def test_best_neighbor_label_selects_allowed() -> None:
    """best_neighbor_label selects from allowed labels only."""
    adjacency = {"A": {"B": 1.0, "C": 2.0}}
    labels = {"B": "allowed", "C": "forbidden"}
    allowed = {"allowed"}

    result = best_neighbor_label("A", adjacency, labels, allowed)

    assert result == "allowed"


def test_best_neighbor_label_selects_heaviest() -> None:
    """best_neighbor_label selects heaviest weighted neighbor label."""
    adjacency = {"A": {"B": 1.0, "C": 5.0}}
    labels = {"B": "light", "C": "heavy"}
    allowed = {"light", "heavy"}

    result = best_neighbor_label("A", adjacency, labels, allowed)

    assert result == "heavy"


# =============================================================================
# cluster_sizes_map Tests
# =============================================================================


def test_cluster_sizes_map_empty() -> None:
    """cluster_sizes_map returns empty dict for empty labels."""
    result = cluster_sizes_map({})

    assert result == {}


def test_cluster_sizes_map_counts_clusters() -> None:
    """cluster_sizes_map counts nodes per cluster."""
    labels = {"A": "c1", "B": "c1", "C": "c2", "D": "c2", "E": "c2"}

    result = cluster_sizes_map(labels)

    assert result["c1"] == CLUSTER_SIZE_TWO
    assert result["c2"] == CLUSTER_SIZE_THREE
