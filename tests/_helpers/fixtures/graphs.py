"""Unified graph fixtures for test helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import networkx as nx

GraphKind = Literal["chain", "star", "cycle", "layered", "golden", "custom"]

_ALPHABET_SIZE: Final[int] = 26


@dataclass(frozen=True)
class GraphFixtureSpec:
    """Specification for building a test graph."""

    kind: GraphKind
    directed: bool = True
    nodes: int | None = None
    edges: int | None = None
    layers: tuple[int, ...] | None = None
    spokes: int | None = None
    cycle_size: int | None = None
    seed: int | None = None


class GraphFixtureFactory:
    """Factory for building standardized NetworkX graphs."""

    @staticmethod
    def build(spec: GraphFixtureSpec) -> nx.Graph:
        """Build a graph from a fixture specification."""
        if spec.kind == "chain":
            return _build_chain(spec)
        if spec.kind == "star":
            return _build_star(spec)
        if spec.kind == "cycle":
            return _build_cycle(spec)
        if spec.kind == "layered":
            return _build_layered(spec)
        if spec.kind == "golden":
            return _build_golden(spec)
        return _build_custom(spec)


STANDARD_CALL: Final[GraphFixtureSpec] = GraphFixtureSpec(kind="chain", directed=True, nodes=4)
STANDARD_IMPORT: Final[GraphFixtureSpec] = GraphFixtureSpec(kind="cycle", directed=True, cycle_size=3)
GOLDEN_CALL: Final[GraphFixtureSpec] = GraphFixtureSpec(kind="golden", directed=True)
GOLDEN_IMPORT: Final[GraphFixtureSpec] = GraphFixtureSpec(kind="golden", directed=True)


def _build_chain(spec: GraphFixtureSpec) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if spec.directed else nx.Graph()
    length = spec.nodes or 0
    if length < 1:
        return graph

    def node_label(i: int) -> str:
        return chr(ord("A") + i % _ALPHABET_SIZE) if i < _ALPHABET_SIZE else f"N{i}"

    nodes = [node_label(i) for i in range(length)]
    graph.add_nodes_from(nodes)
    for i in range(length - 1):
        graph.add_edge(nodes[i], nodes[i + 1])
    return graph


def _build_star(spec: GraphFixtureSpec) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if spec.directed else nx.Graph()
    spokes = spec.spokes or 0
    hub = "hub"
    graph.add_node(hub)
    for idx in range(spokes):
        leaf = f"spoke{idx + 1}"
        graph.add_node(leaf)
        graph.add_edge(hub, leaf)
    return graph


def _build_cycle(spec: GraphFixtureSpec) -> nx.Graph:
    size = spec.cycle_size or spec.nodes or 0
    graph = _build_chain(GraphFixtureSpec(kind="chain", directed=spec.directed, nodes=size))
    nodes = list(graph.nodes())
    if len(nodes) > 1:
        graph.add_edge(nodes[-1], nodes[0])
    return graph


def _build_layered(spec: GraphFixtureSpec) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if spec.directed else nx.Graph()
    layers = spec.layers or (4, 5, 3, 2)
    layer_nodes: list[list[str]] = []
    for layer_index, count in enumerate(layers):
        nodes = [f"L{layer_index}_{idx}" for idx in range(count)]
        graph.add_nodes_from(nodes)
        layer_nodes.append(nodes)
    for idx in range(len(layer_nodes) - 1):
        for node in layer_nodes[idx]:
            for downstream in layer_nodes[idx + 1]:
                graph.add_edge(node, downstream)
    return graph


def _build_golden(spec: GraphFixtureSpec) -> nx.Graph:
    layered = GraphFixtureSpec(kind="layered", directed=spec.directed, layers=(4, 5, 5, 3))
    graph = _build_layered(layered)
    if spec.directed and graph.number_of_nodes() >= 5:
        nodes = list(graph.nodes())
        graph.add_edge(nodes[1], nodes[2])
        graph.add_edge(nodes[2], nodes[1])
    return graph


def _build_custom(spec: GraphFixtureSpec) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if spec.directed else nx.Graph()
    if spec.nodes is not None:
        graph.add_nodes_from(range(spec.nodes))
    return graph


__all__ = [
    "GOLDEN_CALL",
    "GOLDEN_IMPORT",
    "GraphFixtureFactory",
    "GraphFixtureSpec",
    "GraphKind",
    "STANDARD_CALL",
    "STANDARD_IMPORT",
]
