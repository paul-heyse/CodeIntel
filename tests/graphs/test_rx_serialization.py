"""Tests for rustworkx node-link serialization."""

from __future__ import annotations

from pathlib import Path

from codeintel.build.graphs.rx import (
    RxGraphStore,
    decode_node_payload,
    read_node_link_json,
    write_node_link_json,
)
from tests._helpers.assertions import expect_equal


def test_node_link_roundtrip(tmp_path: Path) -> None:
    """Node-link JSON should roundtrip graph structure and payloads."""
    store = RxGraphStore.directed()
    store.set_node_attrs("a", {"kind": "root"})
    store.add_weighted_edge("a", "b", weight=1.0)
    path = tmp_path / "graph.json"
    write_node_link_json(path, store.graph)
    restored = read_node_link_json(path)
    expect_equal(restored.num_nodes(), 2, label="node_count")
    expect_equal(restored.num_edges(), 1, label="edge_count")
    decoded = [
        decode_node_payload(restored.get_node_data(node_idx))
        for node_idx in restored.node_indices()
    ]
    expect_equal({node_id for node_id, _ in decoded}, {"a", "b"}, label="node_payloads")
    attrs_by_id = dict(decoded)
    expect_equal(attrs_by_id.get("a", {}).get("kind"), "root", label="node_attrs")
    edges = restored.edge_list()
    expect_equal(len(edges), 1, label="edge_list_len")
    left, right = edges[0]
    weight = restored.get_edge_data(left, right)
    expect_equal(weight, 1.0, label="edge_weight")
