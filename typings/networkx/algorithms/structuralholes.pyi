from collections.abc import Hashable, Iterable

from networkx import DiGraph, Graph

Node = Hashable

def constraint(
    graph: Graph | DiGraph,
    nodes: Iterable[Node] | None = None,
    *,
    weight: str | None = None,
) -> dict[Node, float]: ...
