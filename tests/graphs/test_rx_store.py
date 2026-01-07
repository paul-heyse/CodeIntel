"""Tests for the rustworkx graph store."""

from __future__ import annotations

from codeintel.build.graphs.rx import RxGraphStore, decode_node_payload
from tests._helpers.assertions import expect_equal, expect_false, expect_true


def test_rx_graph_store_ensures_nodes() -> None:
    """Ensure repeated inserts reuse node indices and maintain ID maps."""
    store = RxGraphStore.directed()
    first = store.ensure_node("alpha")
    second = store.ensure_node("alpha")
    expect_equal(first, second, label="node_index_stable")
    expect_equal(store.graph.num_nodes(), 1, label="node_count")
    expect_equal(store.get_id(first), "alpha", label="id_roundtrip")
    store.ensure_node("beta")
    expect_equal(store.node_ids(), ["alpha", "beta"], label="node_ids_sorted")


def test_rx_graph_store_add_weighted_edge_accumulates() -> None:
    """Edges should aggregate weights instead of adding parallel edges."""
    store = RxGraphStore.directed()
    store.add_weighted_edge("a", "b", weight=1.0)
    store.add_weighted_edge("a", "b", weight=2.5)
    src_idx = store.get_index("a")
    dst_idx = store.get_index("b")
    expect_true(
        src_idx is not None and dst_idx is not None,
        message="Expected node indices to be present after edge insertion.",
    )
    if src_idx is None or dst_idx is None:
        return
    weight = store.graph.get_edge_data(src_idx, dst_idx)
    expect_true(store.graph.num_edges() == 1, message="Expected a single aggregated edge")
    expect_equal(weight, 3.5, label="edge_weight_sum")


def test_rx_graph_store_set_node_attrs_updates_payload() -> None:
    """Node attribute updates should be reflected in payloads."""
    store = RxGraphStore.directed()
    store.set_node_attrs("alpha", {"kind": "function"})
    node_idx = store.get_index("alpha")
    expect_true(node_idx is not None, message="Expected node to be created")
    if node_idx is None:
        return
    payload = store.graph.get_node_data(node_idx)
    node_id, attrs = decode_node_payload(payload)
    expect_equal(node_id, "alpha", label="payload_id")
    expect_equal(attrs.get("kind"), "function", label="payload_attrs")


def test_rx_graph_store_set_edge_weight_updates_payload() -> None:
    """Setting an edge weight should update the existing payload."""
    store = RxGraphStore.directed()
    store.add_weighted_edge("a", "b", weight=1.0)
    updated = store.set_edge_weight("a", "b", weight=2.0)
    expect_true(updated, message="Expected edge weight update to succeed")
    src_idx = store.get_index("a")
    dst_idx = store.get_index("b")
    expect_true(src_idx is not None and dst_idx is not None, message="Expected indices")
    if src_idx is None or dst_idx is None:
        return
    weight = store.graph.get_edge_data(src_idx, dst_idx)
    expect_equal(weight, 2.0, label="edge_weight_update")


def test_rx_graph_store_set_edge_weight_missing_nodes_returns_false() -> None:
    """Setting edge weight should fail for missing nodes."""
    store = RxGraphStore.directed()
    updated = store.set_edge_weight("missing", "node", weight=1.0)
    expect_false(updated, message="Expected missing edge update to return False")
