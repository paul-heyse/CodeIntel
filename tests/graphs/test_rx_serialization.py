"""Tests for rustworkx node-link serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.build.graphs.rx import (
    RxGraphStore,
    decode_node_payload,
    dumps_node_link_json,
    read_node_link_json,
    write_node_link_json,
)
from codeintel.build.graphs.rx.metadata import GraphMetadata, apply_graph_metadata
from tests._helpers.assertions import expect_equal, expect_not_empty


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
    expect_equal({node_id for node_id, _ in decoded}, {None}, label="node_payloads")
    attrs_by_id = dict(decoded)
    expect_equal(attrs_by_id.get("a", {}), {}, label="node_attrs")
    edges = restored.edge_list()
    expect_equal(len(edges), 1, label="edge_list_len")
    left, right = edges[0]
    weight = restored.get_edge_data(left, right)
    expect_equal(weight, None, label="edge_weight")


def test_node_link_requires_metadata_for_cache() -> None:
    """Serialization should enforce metadata requirements when configured."""
    store = RxGraphStore.directed()
    store.add_weighted_edge("a", "b", weight=1.0)

    with pytest.raises(ValueError, match="Graph metadata missing"):
        dumps_node_link_json(store.graph, require_metadata=True)

    metadata = GraphMetadata(
        cache_version="v4",
        engine="rustworkx",
        graph_kind="CALL_GRAPH",
        weight_policy=store.weight_policy.name,
    )
    apply_graph_metadata(store.graph, metadata)
    payload = dumps_node_link_json(store.graph, require_metadata=True)
    expect_not_empty(payload, label="metadata_payload")
