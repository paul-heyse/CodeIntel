from collections.abc import Hashable, Iterable, Sequence

from networkx import Graph

Node = Hashable

def greedy_modularity_communities(
    graph: Graph,
    *,
    weight: str | None = None,
    resolution: float | None = None,
) -> Sequence[set[Node]]: ...
def louvain_communities(
    graph: Graph,
    *,
    weight: str | None = None,
    resolution: float | None = None,
    seed: int | None = None,
) -> Sequence[set[Node]]: ...
def label_propagation_communities(graph: Graph) -> Sequence[set[Node]]: ...
def modularity(
    graph: Graph,
    communities: Iterable[set[Node]],
    *,
    resolution: float | None = None,
    weight: str | None = None,
) -> float: ...
