from collections.abc import Hashable, Iterable

from networkx import Graph

Node = Hashable

def degree_centrality(graph: Graph, nodes: Iterable[Node]) -> dict[Node, float]: ...
def weighted_projected_graph(graph: Graph, nodes: Iterable[Node]) -> Graph: ...
