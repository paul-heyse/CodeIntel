"""Subsystem edge statistics helpers."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class SubsystemEdgeStats:
    """Edge counts and fan-in/out sets for a subsystem."""

    internal_edges: int
    external_edges: int
    fan_in: set[str]
    fan_out: set[str]


def compute_subsystem_edge_stats(
    members: list[str],
    labels: dict[str, str],
    import_graph: nx.DiGraph,
) -> SubsystemEdgeStats:
    """
    Compute edge statistics for a subsystem cluster.

    Returns
    -------
    SubsystemEdgeStats
        Aggregated edge metrics for the subsystem.
    """
    member_set = set(members)
    label = labels.get(members[0]) if members else None
    internal_edges = 0
    external_edges = 0
    fan_in: set[str] = set()
    fan_out: set[str] = set()

    for src, dst, data in import_graph.edges(data=True):
        src_label = labels.get(src)
        dst_label = labels.get(dst)
        if src_label is None or dst_label is None:
            continue
        weight = int(data.get("weight", 1))
        if src in member_set and dst in member_set:
            internal_edges += weight
        elif src_label == label and dst_label != label:
            external_edges += weight
            fan_out.add(dst_label)
        elif dst_label == label and src_label != label:
            external_edges += weight
            fan_in.add(src_label)

    return SubsystemEdgeStats(
        internal_edges=internal_edges,
        external_edges=external_edges,
        fan_in=fan_in,
        fan_out=fan_out,
    )
