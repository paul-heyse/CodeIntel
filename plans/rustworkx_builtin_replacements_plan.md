# Rustworkx Builtin Replacement Scope

## Purpose
Replace bespoke graph algorithms with rustworkx builtins where parity exists, while keeping
deterministic ordering, NaN handling, and existing tolerance policies.

## Shared policies (apply to all items)
- Keep domain ID mapping in `RxGraphStore` and map rustworkx outputs back to IDs.
- Use deterministic ordering via `stable_key` and `sorted_mapping`.
- Use `edge_weight_from_payload` or a constant weight function when needed.
- Preserve NaN and tolerance handling through `_normalize_float_mapping`.

## Scope items

### 1) Degree centrality (total + in/out)
Rustworkx capability: `rx.degree_centrality`, `rx.in_degree_centrality`, `rx.out_degree_centrality`.

Code pattern:
```python
def degree_centrality_by_id(graph: GraphInput) -> dict[Hashable, float]:
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    if store.is_directed:
        directed = _directed_graph(store)
        in_raw = rx.in_degree_centrality(directed)
        out_raw = rx.out_degree_centrality(directed)
        raw = {idx: in_raw[idx] + out_raw.get(idx, 0.0) for idx in in_raw}
    else:
        undirected = _undirected_graph(store)
        raw = rx.degree_centrality(undirected)
    mapped = {store.index_to_id[idx]: value for idx, value in raw.items()}
    return _normalize_float_mapping(mapped, nan_policy="keep")
```

Target files:
- `src/codeintel/build/graphs/rx/algos.py`

### 2) DAG layers (per node)
Rustworkx capability: `rx.layers` or `rx.topological_generations`.

Code pattern:
```python
def topological_layers(graph: GraphInput) -> dict[Any, int]:
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    directed = _directed_graph(store)
    layer_map: dict[Any, int] = {}
    for layer, generation in enumerate(rx.topological_generations(directed)):
        ordered = sorted(generation, key=lambda idx: stable_key(store.index_to_id[idx]))
        for idx in ordered:
            layer_map[store.index_to_id[idx]] = layer
    return sorted_mapping(layer_map)
```

Target files:
- `src/codeintel/build/graphs/compute/metrics/components.py`

### 3) Condensation layer count (DAG longest path)
Rustworkx capability: `rx.dag_longest_path_length` (or `rx.layers` count).

Code pattern:
```python
def _layer_count(graph: rx.PyDiGraph) -> int:
    if graph.num_nodes() == 0:
        return 0
    return int(rx.dag_longest_path_length(graph)) + 1
```

Target files:
- `src/codeintel/build/graphs/compute/metrics/statistics.py`

### 4) Average shortest path length (largest component)
Rustworkx capability: `rx.graph_unweighted_average_shortest_path_length`.

Code pattern:
```python
def compute_avg_shortest_path_length(graph: GraphInput) -> float | None:
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return None
    work_store = to_undirected_store(store)
    largest = _largest_component(work_store)
    if largest is None:
        return None
    if len(largest) <= 1:
        return 0.0
    subgraph = _undirected_graph(work_store).subgraph(list(largest), preserve_attrs=True)
    return float(rx.graph_unweighted_average_shortest_path_length(subgraph))
```

Target files:
- `src/codeintel/build/graphs/compute/metrics/statistics.py`

### 5) Diameter estimate (largest component)
Rustworkx capability: `rx.graph_distance_matrix`.

Code pattern:
```python
import math

def compute_diameter_estimate(graph: GraphInput) -> float | None:
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return None
    work_store = to_undirected_store(store)
    largest = _largest_component(work_store)
    if largest is None:
        return None
    if len(largest) <= 1:
        return 0.0
    subgraph = _undirected_graph(work_store).subgraph(list(largest), preserve_attrs=True)
    matrix = rx.graph_distance_matrix(subgraph)
    max_distance = max(
        distance
        for row in matrix
        for distance in row
        if math.isfinite(distance)
    )
    return float(max_distance)
```

Target files:
- `src/codeintel/build/graphs/compute/metrics/statistics.py`

### 6) DFG path length stats (max, avg, reach)
Rustworkx capability: `rx.digraph_dijkstra_shortest_path_lengths`
or `rx.digraph_all_pairs_dijkstra_path_lengths`.

Code pattern:
```python
def compute_dfg_path_lengths(
    graph: GraphInput,
    *,
    max_depth: int = 100,
) -> dict[Any, DFGPathStats]:
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    directed = cast("rx.PyDiGraph", store.graph)
    weight_fn = lambda _payload: 1.0
    result: dict[Any, DFGPathStats] = {}
    for node_id in store.node_ids():
        node_idx = store.id_to_index[node_id]
        raw = rx.digraph_dijkstra_shortest_path_lengths(directed, node_idx, weight_fn)
        bounded = [dist for dist in raw.values() if 0 < dist <= max_depth]
        if bounded:
            result[node_id] = DFGPathStats(
                max_def_use_distance=max(bounded),
                avg_def_use_distance=sum(bounded) / len(bounded),
                reach_count=len(bounded),
            )
        else:
            result[node_id] = DFGPathStats(
                max_def_use_distance=0,
                avg_def_use_distance=0.0,
                reach_count=0,
            )
    return result
```

Target files:
- `src/codeintel/build/graphs/compute/metrics/dfg.py`

### 7) Graph fixture generators
Rustworkx capability: `rustworkx.generators.*` (path, cycle, star, complete, barbell).

Code pattern:
```python
def _store_from_generator(
    graph: rx.PyGraph | rx.PyDiGraph,
    *,
    label_prefix: str = "N",
) -> RxGraphStore:
    store = (
        RxGraphStore.directed()
        if isinstance(graph, rx.PyDiGraph)
        else RxGraphStore.undirected()
    )
    index_to_id = {idx: f"{label_prefix}{idx}" for idx in graph.node_indices()}
    for node_id in index_to_id.values():
        store.ensure_node(node_id)
    for src_idx, dst_idx in graph.edge_list():
        store.add_weighted_edge(index_to_id[src_idx], index_to_id[dst_idx], weight=1.0)
    return store

def chain_graph(length: int) -> RxGraphStore:
    raw = rx.generators.directed_path_graph(length)
    return _store_from_generator(raw, label_prefix="n")
```

Target files:
- `tests/_helpers/fixtures/graphs.py`
