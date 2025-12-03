"""Pure component analysis functions.

This module provides stateless functions for computing graph components
and structural properties without any database or file I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import networkx as nx


@dataclass(frozen=True)
class ComponentInfo:
    """Information about a connected component.

    Attributes
    ----------
    component_id
        Component identifier.
    size
        Number of nodes in component.
    nodes
        Nodes in the component.
    """

    component_id: int
    size: int
    nodes: frozenset[Any]


@dataclass(frozen=True)
class SCCResult:
    """Result of strongly connected component analysis.

    Attributes
    ----------
    components
        List of SCCs.
    node_to_component
        Node to component ID mapping.
    condensation
        DAG of condensed SCCs (None if not computed).
    """

    components: tuple[ComponentInfo, ...]
    node_to_component: dict[Any, int]
    condensation: nx.DiGraph | None = None


class ComponentStats(TypedDict):
    """Summary statistics for a collection of components."""

    count: int
    largest_size: int
    smallest_size: int
    mean_size: float
    singleton_count: int


def find_strongly_connected(
    graph: nx.DiGraph,
    *,
    compute_condensation: bool = False,
) -> SCCResult:
    """Find strongly connected components in a directed graph.

    Parameters
    ----------
    graph
        Directed graph.
    compute_condensation
        Whether to compute the condensation DAG.

    Returns
    -------
    SCCResult
        SCC analysis result.

    Examples
    --------
    >>> g = nx.DiGraph([(1, 2), (2, 3), (3, 1), (4, 5)])
    >>> result = find_strongly_connected(g)
    >>> len(result.components) >= 2
    True
    """
    if graph.number_of_nodes() == 0:
        return SCCResult(components=(), node_to_component={})

    sccs = list(nx.strongly_connected_components(graph))
    components: list[ComponentInfo] = []
    node_to_component: dict[Any, int] = {}

    for idx, scc in enumerate(sccs):
        nodes_frozen: frozenset[Any] = frozenset(scc)
        components.append(
            ComponentInfo(
                component_id=idx,
                size=len(scc),
                nodes=nodes_frozen,
            )
        )
        for node in scc:
            node_to_component[node] = idx

    condensation = None
    if compute_condensation:
        condensation = nx.condensation(graph, scc=sccs)

    return SCCResult(
        components=tuple(components),
        node_to_component=node_to_component,
        condensation=condensation,
    )


def find_weakly_connected(graph: nx.DiGraph) -> list[ComponentInfo]:
    """Find weakly connected components in a directed graph.

    Parameters
    ----------
    graph
        Directed graph.

    Returns
    -------
    list[ComponentInfo]
        Weakly connected components.
    """
    if graph.number_of_nodes() == 0:
        return []

    wccs = list(nx.weakly_connected_components(graph))
    return [
        ComponentInfo(
            component_id=idx,
            size=len(wcc),
            nodes=frozenset(wcc),
        )
        for idx, wcc in enumerate(wccs)
    ]


def find_connected(graph: nx.Graph) -> list[ComponentInfo]:
    """Find connected components in an undirected graph.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    list[ComponentInfo]
        Connected components.
    """
    if graph.number_of_nodes() == 0:
        return []

    ccs = list(nx.connected_components(graph))
    return [
        ComponentInfo(
            component_id=idx,
            size=len(cc),
            nodes=frozenset(cc),
        )
        for idx, cc in enumerate(ccs)
    ]


def find_bridges(graph: nx.Graph) -> list[tuple[Any, Any]]:
    """Find bridge edges whose removal disconnects the graph.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    list[tuple[Any, Any]]
        Bridge edges.
    """
    if graph.number_of_nodes() == 0:
        return []
    return list(nx.bridges(graph))


def find_articulation_points(graph: nx.Graph) -> list[Any]:
    """Find articulation points whose removal disconnects the graph.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    list[Any]
        Articulation point nodes.
    """
    if graph.number_of_nodes() == 0:
        return []
    return list(nx.articulation_points(graph))


def compute_component_stats(
    components: Sequence[ComponentInfo],
) -> ComponentStats:
    """Compute summary statistics for components.

    Parameters
    ----------
    components
        Component information.

    Returns
    -------
    ComponentStats
        Statistics including count, sizes, and largest component.
    """
    if not components:
        return ComponentStats(
            count=0,
            largest_size=0,
            smallest_size=0,
            mean_size=0.0,
            singleton_count=0,
        )

    sizes = [c.size for c in components]
    return ComponentStats(
        count=len(components),
        largest_size=max(sizes),
        smallest_size=min(sizes),
        mean_size=sum(sizes) / len(sizes),
        singleton_count=sum(1 for s in sizes if s == 1),
    )


def find_cycles(graph: nx.DiGraph, limit: int | None = 100) -> list[list[Any]]:
    """Find simple cycles in a directed graph.

    Parameters
    ----------
    graph
        Directed graph.
    limit
        Maximum number of cycles to return (None for all).

    Returns
    -------
    list[list[Any]]
        List of cycles as node lists.
    """
    if graph.number_of_nodes() == 0:
        return []

    cycles: list[list[Any]] = []
    for cycle in nx.simple_cycles(graph):
        cycles.append(cycle)
        if limit is not None and len(cycles) >= limit:
            break
    return cycles


def topological_layers(graph: nx.DiGraph) -> dict[Any, int]:
    """Compute topological layer for each node in a DAG.

    Parameters
    ----------
    graph
        Directed acyclic graph.

    Returns
    -------
    dict[Any, int]
        Node to layer mapping (0 for roots).

    Notes
    -----
    If the graph contains cycles, NetworkX will raise `nx.NetworkXUnfeasible`.
    """
    if graph.number_of_nodes() == 0:
        return {}

    layers: dict[Any, int] = {node: 0 for node in graph.nodes() if graph.in_degree(node) == 0}
    for node in nx.topological_sort(graph):
        base = layers.get(node, 0)
        for succ in graph.successors(node):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return layers


def condensation_layers(
    graph: nx.DiGraph,
    scc_result: SCCResult,
) -> dict[Any, int]:
    """Compute layers based on SCC condensation.

    Parameters
    ----------
    graph
        Original directed graph.
    scc_result
        SCC analysis result with condensation.

    Returns
    -------
    dict[Any, int]
        Node to layer mapping based on condensation.
    """
    if scc_result.condensation is None:
        return {}

    condensation = scc_result.condensation
    if condensation.number_of_nodes() == 0:
        return {}

    comp_layers = topological_layers(condensation)
    return {
        node: comp_layers.get(scc_result.node_to_component.get(node, -1), 0)
        for node in graph.nodes()
    }


__all__ = [
    "ComponentInfo",
    "ComponentStats",
    "SCCResult",
    "compute_component_stats",
    "condensation_layers",
    "find_articulation_points",
    "find_bridges",
    "find_connected",
    "find_cycles",
    "find_strongly_connected",
    "find_weakly_connected",
    "topological_layers",
]
