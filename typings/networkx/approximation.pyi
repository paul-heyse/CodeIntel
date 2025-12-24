from collections.abc import Hashable

from networkx import DiGraph, Graph

Node = Hashable

def diameter(graph: Graph | DiGraph) -> int: ...
