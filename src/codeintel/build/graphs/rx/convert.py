"""Conversion helpers between rustworkx graphs and stores."""

from __future__ import annotations

import rustworkx as rx

from codeintel.build.graphs.rx.policies import GraphNumericPolicy, GraphWeightPolicy
from codeintel.build.graphs.rx.store import RxGraphStore

RxGraph = rx.PyGraph | rx.PyDiGraph


def store_from_rx(
    graph: RxGraph,
    *,
    weight_policy: GraphWeightPolicy | None = None,
    numeric_policy: GraphNumericPolicy | None = None,
) -> RxGraphStore:
    """Convert a rustworkx graph into a rustworkx-backed store.

    Returns
    -------
    RxGraphStore
        Rustworkx graph store populated from the rustworkx payloads.
    """
    return RxGraphStore.from_rx_graph(
        graph,
        weight_policy=weight_policy,
        numeric_policy=numeric_policy,
    )


__all__ = ["RxGraph", "store_from_rx"]
