"""Shared component helpers for rustworkx graphs."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence

from codeintel.build.graphs.rx.normalize import stable_key
from codeintel.build.graphs.rx.store import RxGraphStore


def component_sort_key(store: RxGraphStore, component: Iterable[int]) -> tuple[str, str]:
    """Return a deterministic sort key for a component.

    Returns
    -------
    tuple[str, str]
        Stable sort key for component ordering.
    """
    members = list(component)
    if not members:
        return ("", "")
    smallest = min((store.index_to_id[idx] for idx in members), key=stable_key)
    return stable_key(smallest)


def sort_components(store: RxGraphStore, components: Iterable[Iterable[int]]) -> list[set[int]]:
    """Return components sorted by the stable node-id key.

    Returns
    -------
    list[set[int]]
        Components sorted by node-id stable key.
    """
    normalized = [set(component) for component in components]
    return sorted(normalized, key=lambda comp: component_sort_key(store, comp))


def component_membership(components: Sequence[set[int]]) -> dict[int, int]:
    """Return node-index membership mapping for sorted components.

    Returns
    -------
    dict[int, int]
        Mapping of node index to component id.
    """
    mapping: dict[int, int] = {}
    for comp_id, component in enumerate(components):
        for node_idx in component:
            mapping[node_idx] = comp_id
    return mapping


def component_membership_by_id(
    store: RxGraphStore,
    components: Sequence[set[int]],
) -> dict[Hashable, int]:
    """Return node-id membership mapping for sorted components.

    Returns
    -------
    dict[Hashable, int]
        Mapping of node id to component id.
    """
    index_membership = component_membership(components)
    return {store.index_to_id[idx]: comp_id for idx, comp_id in index_membership.items()}


def invert_membership_map(
    membership: Mapping[Hashable, int],
) -> list[list[Hashable]]:
    """Return component member lists ordered by component id.

    Returns
    -------
    list[list[Hashable]]
        Member lists ordered by component id.
    """
    buckets: dict[int, list[Hashable]] = {}
    for node_id, comp_id in membership.items():
        buckets.setdefault(comp_id, []).append(node_id)
    return [sorted(buckets[comp_id], key=stable_key) for comp_id in sorted(buckets)]


__all__ = [
    "component_membership",
    "component_membership_by_id",
    "component_sort_key",
    "invert_membership_map",
    "sort_components",
]
