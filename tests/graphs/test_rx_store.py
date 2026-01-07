"""Tests for the rustworkx graph store."""

from __future__ import annotations

from codeintel.build.graphs.rx import RxGraphStore
from tests._helpers.assertions import expect_equal, expect_true


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
