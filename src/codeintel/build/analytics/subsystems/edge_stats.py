"""Subsystem edge statistics helpers."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.build.graphs.rx.algos import GraphInput, ensure_store


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
    import_graph: GraphInput,
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

    store = ensure_store(import_graph)
    for src_idx, dst_idx in store.graph.edge_list():
        src_key = str(store.index_to_id[src_idx])
        dst_key = str(store.index_to_id[dst_idx])
        src_label = labels.get(src_key)
        dst_label = labels.get(dst_key)
        if src_label is None or dst_label is None:
            continue
        payload = store.graph.get_edge_data(src_idx, dst_idx)
        weight = _coerce_edge_weight(payload)
        if src_key in member_set and dst_key in member_set:
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


def _coerce_edge_weight(value: object) -> int:
    if value is None:
        return 1
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 1
    return 1
