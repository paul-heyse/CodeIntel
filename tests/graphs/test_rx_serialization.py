"""Tests for rustworkx node-link serialization."""

from __future__ import annotations

from pathlib import Path

from codeintel.build.graphs.rx import (
    RxGraphStore,
    read_node_link_json,
    write_node_link_json,
)
from tests._helpers.assertions import expect_equal


def test_node_link_roundtrip(tmp_path: Path) -> None:
    """Node-link JSON should roundtrip graph structure and payloads."""
    store = RxGraphStore.directed()
    store.add_weighted_edge("a", "b", weight=1.0)
    path = tmp_path / "graph.json"
    write_node_link_json(path, store.graph)
    restored = read_node_link_json(path)
    expect_equal(restored.num_nodes(), 2, label="node_count")
    expect_equal(restored.num_edges(), 1, label="edge_count")
    expect_equal(set(restored.nodes()), {"a", "b"}, label="node_payloads")
    edges = restored.edge_list()
    expect_equal(len(edges), 1, label="edge_list_len")
    left, right = edges[0]
    weight = restored.get_edge_data(left, right)
    expect_equal(weight, 1.0, label="edge_weight")
