# Rustworkx Migration Plan

## Goals
- Replace all NetworkX usage with rustworkx across build, analytics, and tests.
- Preserve current graph semantics, metrics, and output schemas.
- Improve performance and determinism with index-based graphs and typed wrappers.

## Non-Goals
- Add a GPU backend (rustworkx is CPU-only).
- Change table schemas, analytics outputs, or data model contracts.
- Introduce new graph algorithms beyond NetworkX parity.

## Scope Summary
- Graph engine and runtime:
  - `src/codeintel/build/graphs/engine/nx_engine.py`
  - `src/codeintel/build/graphs/engine/views.py`
  - `src/codeintel/build/graphs/engine/protocol.py`
  - `src/codeintel/build/graphs/runtime/runtime.py`
  - `src/codeintel/core/resources/graphs.py`
- Graph construction and adapters:
  - `src/codeintel/build/graphs/builders.py`
  - `src/codeintel/build/graphs/compute/imports.py`
  - `src/codeintel/build/graphs/ports/engine.py`
  - `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
  - `src/codeintel/build/hamilton/native/analytics/config_graphs.py`
- Algorithm layers:
  - `src/codeintel/core/compute/centrality.py`
  - `src/codeintel/build/graphs/compute/metrics/*`
- Analytics orchestration and consumers:
  - `src/codeintel/build/analytics/graphs/*`
  - `src/codeintel/build/analytics/subsystems/*`
  - `src/codeintel/build/analytics/cfg_dfg/*`
  - `src/codeintel/build/analytics/functions/function_effects.py`
- Validation:
  - `src/codeintel/build/graphs/validation/*`
- Tests and fixtures:
  - `tests/_helpers/fixtures/graphs.py`
  - `tests/analytics/*`
  - `tests/graphs/*`
- Dependency cleanup:
  - `pyproject.toml`, `uv.lock`, `typings/networkx/*`

## Rustworkx Constraints and Design Implications
- Nodes and edges are integer indices; indices can be reused after deletions.
  We will avoid deletions and treat indices as ephemeral handles.
- Node and edge payloads are arbitrary Python objects.
  Weight-aware algorithms require `weight_fn(edge_payload) -> float`.
- Multigraph support is optional. We will use `multigraph=False` and aggregate
  multiplicity into edge weight to preserve existing semantics.
- Many rustworkx algorithms return custom containers (`CentralityMapping`,
  `PathMapping`, `NodeIndices`, `NodeMap`) with non-deterministic iteration in
  some cases; we will normalize to plain dict/list and sort keys.
- Several algorithms are unweighted in rustworkx (betweenness, closeness).
  Weighted variants are separate functions or absent; we will add custom
  implementations where parity is required.
- PyDiGraph cycle checking is optional and expensive; we will keep
  `check_cycle=False` and compute cycles via algorithms when needed.

## Target Architecture

### 1) Graph Identity and Payload Policy
- Node identity is a stable domain ID (e.g., GOID, module string).
- rustworkx uses integer indices; we maintain an explicit mapping.
- Node payloads store the domain ID for easy reverse mapping.
- Edge payloads store a numeric weight (int or float).
- Graphs are immutable after build (no node removals) to avoid index reuse.

### 2) Graph Store Abstractions
Introduce a rustworkx-first graph store that hides index management and
returns domain-ID keyed results to analytics callers.

Proposed new module:
- `src/codeintel/build/graphs/rx/store.py`
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/rx/serialization.py`

### 3) Algorithm Wrappers
Wrap rustworkx algorithms to:
- Accept domain IDs.
- Convert to indices.
- Normalize rustworkx return types to dict/list with deterministic ordering.
- Apply weight_fn policies consistently.
- Surface consistent error envelopes mapped from rustworkx exceptions.

### 4) Serialization and Cache
Replace NetworkX node-link serialization with rustworkx node-link JSON.
Cache metadata remains unchanged, but on-disk graph format changes.

### 5) Return Type Normalization
Normalize rustworkx return types into deterministic Python containers:
- `CentralityMapping` -> `dict[node_id, float]` sorted by node_id.
- `PathMapping` -> `dict[target_id, list[node_id]]` sorted by target_id.
- `AllPairsPathMapping` -> nested dicts sorted by source/target.
- `NodeIndices` / `EdgeList` -> `list[...]` sorted when used for output.
- `NodeMap` / `ProductNodeMap` -> `dict[...]` sorted by key.

### 6) Backend Configuration
Remove nx-cugraph backend selection and simplify GraphBackendConfig to a
rustworkx-only execution model.

## Aggressive Migration Plan (Big-Bang Cutover)

### Current Baseline (Implemented)
- Added `GraphBackendConfig.engine` and cache versioning/engine metadata.
- Added rustworkx foundation modules:
  - `src/codeintel/build/graphs/rx/store.py`
  - `src/codeintel/build/graphs/rx/normalize.py`
  - `src/codeintel/build/graphs/rx/serialization.py`
  - `src/codeintel/build/graphs/rx/errors.py`
  - `src/codeintel/build/graphs/rx/payloads.py`
  - `src/codeintel/build/graphs/rx/convert.py`
- Updated `RxGraphStore` to persist node attributes in payloads.
- Replaced NetworkX builders in `src/codeintel/build/graphs/builders.py` with rustworkx stores,
  converting to NetworkX only at the API boundary.
- Converted loaders in `src/codeintel/build/graphs/engine/views.py` to build rustworkx graphs
  and return NetworkX graphs via conversion.
- Replaced the compatibility shim with a real `RxGraphEngine` and made the factory ignore
  GPU/backend preferences (rustworkx-only execution).
- Switched graph cache read/write to rustworkx node-link JSON (`GRAPH_CACHE_VERSION = "v3"`).
- Updated tests for rustworkx store/serialization and engine/view parity:
  - `tests/graphs/test_rx_store.py`
  - `tests/graphs/test_rx_normalize.py`
  - `tests/graphs/test_rx_serialization.py`
  - `tests/graphs/test_engine_nx.py`
  - `tests/graphs/test_engine_factory.py`
- Completed Workstream B algorithm + metrics migration:
  - `src/codeintel/build/graphs/rx/algos.py` wrappers with deterministic ordering, NaN policy,
    and tolerance handling (pagerank, betweenness, closeness, eigenvector, harmonic,
    clustering/triangles, core numbers, constraint/effective size, bipartite projection).
  - `src/codeintel/core/compute/centrality.py` switched to rustworkx wrappers and store inputs.
  - `src/codeintel/build/graphs/compute/metrics/*` moved to rustworkx/custom implementations
    (`centrality.py`, `structural.py`, `paths.py`, `dfg.py`, `cfg.py`, `components.py`,
    `statistics.py`, `community.py`, `bipartite.py`, `projections.py`).
  - Deterministic ordering, NaN handling, and numeric tolerance policies applied across outputs.

### Workstreams (Run in Parallel; No Gradual Rollout)

Workstream A: Engine + Graph Construction (Complete Replacement)
- Status: Implemented (see Current Baseline).
- Compatibility boundary remains: analytics still consume NetworkX graphs while
  rustworkx is the internal construction engine.

Workstream B: Algorithm + Metrics Migration (Parity First)
- Status: Implemented.
- Rustworkx-first wrappers and metric modules now provide deterministic outputs, NaN handling,
  and numeric tolerance policies for custom algorithms.

Workstream C: Analytics + Validation + Tests (Full Cutover)
- Update analytics orchestrators and subsystem metrics to consume rustworkx.
- Migrate validation checks to rustworkx-compatible operations.
- Replace NetworkX fixtures/tests and remove `typings/networkx`.

### Next Actions (Recommended, Two Sets)

Set 1 — Workstream C (Analytics + Validation Cutover)
- Update analytics consumers (`build/analytics/graphs/*`, `build/analytics/subsystems/*`,
  `build/analytics/cfg_dfg/*`, `build/analytics/functions/function_effects.py`) to call
  rustworkx wrappers or `GraphInput`-based helpers directly.
- Migrate validation checks and fixtures under `src/codeintel/build/graphs/validation/*`
  to rustworkx-compatible operations and deterministic outputs.
- Update analytics tests/fixtures to expect rustworkx-driven deterministic ordering and
  the new NaN/tolerance policies.

Set 2 — Dependency Removal + API Surface Cleanup
- Remove NetworkX dependency/stubs from `pyproject.toml`, `uv.lock`,
  and `typings/networkx`, then update type annotations to rustworkx-first types.
- Simplify or remove NetworkX-compatible protocols/adapters in
  `src/codeintel/build/graphs/engine/*`, `src/codeintel/build/graphs/runtime/*`,
  and `src/codeintel/core/resources/graphs.py` if still present for compatibility.
- Update documentation and acceptance checks to reflect rustworkx-only usage.

### Cutover Checklist (Single Switchover)
- Delete the compatibility shim and engine flag; rustworkx only.
- Remove `networkx` dependency and stubs from `pyproject.toml` / `uv.lock`.
- Run quality report and segmented pytest; verify metrics tables unchanged.
- Document rustworkx API usage and deterministic policies.

### Acceptance (Big-Bang)
- No NetworkX imports remain under `src/` or `tests/`.
- Graph metrics tables match current schemas and row counts.
- All tests pass using rustworkx exclusively.

## Algorithm Mapping

### Core Algorithms and Traversal

| NetworkX Usage | Rustworkx Equivalent | Parity Notes |
| --- | --- | --- |
| `nx.pagerank` | `rustworkx.pagerank` | Requires `weight_fn`; parallel edges are summed. |
| `nx.betweenness_centrality` | `rustworkx.betweenness_centrality` | Unweighted only; add custom weighted variant. |
| `nx.closeness_centrality` | `rustworkx.closeness_centrality` | Unweighted; use weighted Newman variant when needed. |
| `nx.eigenvector_centrality` | `rustworkx.eigenvector_centrality` | Uses power iteration with `max_iter`. |
| `nx.harmonic_centrality` | Custom | Implement from shortest path lengths. |
| `nx.descendants` / `nx.ancestors` | `rustworkx.descendants` / `rustworkx.ancestors` | Operate on indices; map back to IDs. |
| `nx.all_simple_paths` | `rustworkx.digraph_all_simple_paths` | Directed only; map indices to IDs. |
| `nx.single_source_shortest_path_length` | `rustworkx.dijkstra_shortest_path_lengths` | Use `weight_fn` and default weight. |
| `nx.all_pairs_shortest_path_length` | `rustworkx.all_pairs_dijkstra_path_lengths` | Consider caps for large graphs. |

### Components, Structure, and Transforms

| NetworkX Usage | Rustworkx Equivalent | Parity Notes |
| --- | --- | --- |
| `nx.strongly_connected_components` | `rustworkx.strongly_connected_components` | Returns `NodeIndices`; normalize. |
| `nx.weakly_connected_components` | `rustworkx.weakly_connected_components` | Returns `NodeIndices`; normalize. |
| `nx.connected_components` | `rustworkx.connected_components` | Use for undirected graphs. |
| `nx.condensation` | Custom | Build from SCC output and edge list. |
| `nx.topological_sort` | `rustworkx.topological_sort` | Raises on cycles; use `TopologicalSorter` if needed. |
| `nx.is_directed_acyclic_graph` | `rustworkx.is_directed_acyclic_graph` | Direct parity. |
| `nx.simple_cycles` | `rustworkx.simple_cycles` | Returns cycles as index lists. |
| `nx.bridges` | `rustworkx.bridges` | Use undirected graph. |
| `nx.articulation_points` | `rustworkx.articulation_points` | Use undirected graph. |
| `nx.dag_longest_path_length` | `rustworkx.dag_longest_path_length` | Weighted variant available. |
| `nx.clustering` | Custom | No rustworkx equivalent. |
| `nx.triangles` | Custom | No rustworkx equivalent. |
| `nx.core_number` | `rustworkx.graph_core_number` / `digraph_core_number` | Use directed or undirected. |
| `nx.constraint` / `nx.effective_size` | Custom | No rustworkx equivalent. |

### Community and Bipartite

| NetworkX Usage | Rustworkx Equivalent | Parity Notes |
| --- | --- | --- |
| `nx.community.*` | Custom | Keep internal implementations. |
| `nx.bipartite.degree_centrality` | Custom | Compute from degree counts. |
| `nx.bipartite.weighted_projected_graph` | Custom | Build projection with weights. |

### Serialization and I/O

| NetworkX Usage | Rustworkx Equivalent | Parity Notes |
| --- | --- | --- |
| `networkx.readwrite.json_graph` | `rustworkx.node_link_json` | Replace cache format. |
| `json_graph.node_link_graph` | `rustworkx.parse_node_link_json` | Replace cache reader. |

## Custom Algorithm Designs

This section specifies the custom algorithms needed to reach parity with
NetworkX behavior, using rustworkx graph primitives and strict normalization
of return types. Each design returns results keyed by domain IDs (not indices).

### Harmonic Centrality

Design:
- Use `rustworkx.dijkstra_shortest_path_lengths` per node (directed or undirected).
- Sum reciprocal distances for reachable nodes only.
- Default is unnormalized (matches NetworkX default).
- Optional `normalized=True` divides by `(n - 1)` when `n > 1`.

```python
import rustworkx as rx


def harmonic_centrality(
    store: RxGraphStore,
    *,
    weight_fn: callable | None = None,
    default_weight: float = 1.0,
    normalized: bool = False,
) -> dict[Hashable, float]:
    graph = store.graph
    node_count = graph.num_nodes()
    results: dict[Hashable, float] = {}
    for node_idx in graph.node_indices():
        lengths = rx.dijkstra_shortest_path_lengths(
            graph,
            node_idx,
            weight_fn=weight_fn,
            default_weight=default_weight,
        )
        total = 0.0
        for target_idx, dist in lengths.items():
            if target_idx == node_idx or dist <= 0:
                continue
            total += 1.0 / float(dist)
        if normalized and node_count > 1:
            total /= float(node_count - 1)
        results[store.index_to_id[node_idx]] = total
    return results
```

### Triangles and Clustering (Undirected)

Design:
- Build undirected neighbor sets from the rustworkx edge list.
- Ignore self-loops.
- Triangles per node are counted by neighbor intersections, divide by 2.
- Clustering coefficient uses:
  - Unweighted: `2 * T / (deg * (deg - 1))`
  - Weighted: geometric mean of normalized weights
    `sum((w_uv * w_uw * w_vw)^(1/3)) / (deg * (deg - 1))`
  - Normalize weights by global max weight, matching NetworkX.

```python
from collections import defaultdict


def _undirected_neighbors(
    graph: rx.PyGraph,
) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left, right in graph.edge_list():
        if left == right:
            continue
        neighbors[left].add(right)
        neighbors[right].add(left)
    return neighbors


def triangles_by_id(store: RxGraphStore) -> dict[Hashable, int]:
    graph = store.graph
    neighbors = _undirected_neighbors(graph)
    counts: dict[int, int] = {}
    for node_idx in graph.node_indices():
        total = 0
        for nbr in neighbors.get(node_idx, set()):
            total += len(neighbors.get(node_idx, set()) & neighbors.get(nbr, set()))
        counts[node_idx] = total // 2
    return {store.index_to_id[idx]: count for idx, count in counts.items()}


def clustering_by_id(
    store: RxGraphStore,
    *,
    weight_fn: callable | None = None,
) -> dict[Hashable, float]:
    graph = store.graph
    neighbors = _undirected_neighbors(graph)
    max_weight = 1.0
    if weight_fn is not None:
        for left, right in graph.edge_list():
            payload = graph.get_edge_data(left, right)
            max_weight = max(max_weight, float(weight_fn(payload)))

    def normalized_weight(left: int, right: int) -> float:
        if weight_fn is None:
            return 1.0
        payload = graph.get_edge_data(left, right)
        return float(weight_fn(payload)) / max_weight

    result: dict[Hashable, float] = {}
    for node_idx in graph.node_indices():
        degree = len(neighbors.get(node_idx, set()))
        if degree < 2:
            result[store.index_to_id[node_idx]] = 0.0
            continue
        if weight_fn is None:
            tri = 0
            for nbr in neighbors[node_idx]:
                tri += len(neighbors[node_idx] & neighbors.get(nbr, set()))
            tri = tri // 2
            result[store.index_to_id[node_idx]] = (2.0 * tri) / (degree * (degree - 1))
            continue

        acc = 0.0
        for v in neighbors[node_idx]:
            for w in neighbors[node_idx]:
                if v >= w:
                    continue
                if w not in neighbors.get(v, set()):
                    continue
                acc += (
                    normalized_weight(node_idx, v)
                    * normalized_weight(node_idx, w)
                    * normalized_weight(v, w)
                ) ** (1.0 / 3.0)
        result[store.index_to_id[node_idx]] = acc / (degree * (degree - 1))
    return result
```

### Constraint and Effective Size (Structural Holes)

Design:
- Use mutual weights: `w_uv + w_vu` for directed graphs.
- Normalized mutual weight:
  - `p_uv = w_uv / sum_k w_uk` for constraint.
  - `m_vw = w_vw / max_k w_vk` for effective size.
- Isolated nodes (including only self-loops) return `NaN`.
- For unweighted, undirected graphs, use the simplified formula
  `e(u) = n - (2t / n)` where `t` is the number of edges among neighbors.

```python
import math


def _mutual_weight(
    graph: rx.PyGraph,
    left: int,
    right: int,
    *,
    weight_fn: callable | None,
) -> float:
    if left == right:
        return 0.0
    if graph.has_edge(left, right):
        payload = graph.get_edge_data(left, right)
        return float(weight_fn(payload)) if weight_fn is not None else 1.0
    return 0.0


def _normalized_mutual_weight(
    graph: rx.PyGraph,
    left: int,
    right: int,
    *,
    weight_fn: callable | None,
    norm: callable,
    neighbors: dict[int, set[int]],
) -> float:
    weights = [
        _mutual_weight(graph, left, nbr, weight_fn=weight_fn) for nbr in neighbors.get(left, set())
    ]
    denom = norm(weights) if weights else 0.0
    if denom == 0.0:
        return 0.0
    return _mutual_weight(graph, left, right, weight_fn=weight_fn) / float(denom)


def constraint_by_id(
    store: RxGraphStore,
    *,
    weight_fn: callable | None = None,
) -> dict[Hashable, float]:
    graph = store.graph
    neighbors = _undirected_neighbors(graph)
    result: dict[Hashable, float] = {}
    for node_idx in graph.node_indices():
        if not neighbors.get(node_idx):
            result[store.index_to_id[node_idx]] = float("nan")
            continue
        total = 0.0
        for nbr in neighbors[node_idx]:
            direct = _normalized_mutual_weight(
                graph,
                node_idx,
                nbr,
                weight_fn=weight_fn,
                norm=sum,
                neighbors=neighbors,
            )
            indirect = 0.0
            for mid in neighbors[node_idx]:
                indirect += (
                    _normalized_mutual_weight(
                        graph,
                        node_idx,
                        mid,
                        weight_fn=weight_fn,
                        norm=sum,
                        neighbors=neighbors,
                    )
                    * _normalized_mutual_weight(
                        graph,
                        mid,
                        nbr,
                        weight_fn=weight_fn,
                        norm=sum,
                        neighbors=neighbors,
                    )
                )
            total += (direct + indirect) ** 2
        result[store.index_to_id[node_idx]] = total
    return result


def effective_size_by_id(
    store: RxGraphStore,
    *,
    weight_fn: callable | None = None,
) -> dict[Hashable, float]:
    graph = store.graph
    neighbors = _undirected_neighbors(graph)
    result: dict[Hashable, float] = {}
    if weight_fn is None:
        for node_idx in graph.node_indices():
            if not neighbors.get(node_idx):
                result[store.index_to_id[node_idx]] = float("nan")
                continue
            ego = neighbors[node_idx]
            tie_count = 0
            for left in ego:
                for right in ego:
                    if left < right and right in neighbors.get(left, set()):
                        tie_count += 1
            n = len(ego)
            result[store.index_to_id[node_idx]] = n - (2.0 * tie_count) / float(n)
        return result

    for node_idx in graph.node_indices():
        if not neighbors.get(node_idx):
            result[store.index_to_id[node_idx]] = float("nan")
            continue
        total = 0.0
        for nbr in neighbors[node_idx]:
            redundancy = 0.0
            for mid in neighbors[node_idx]:
                redundancy += (
                    _normalized_mutual_weight(
                        graph,
                        node_idx,
                        mid,
                        weight_fn=weight_fn,
                        norm=sum,
                        neighbors=neighbors,
                    )
                    * _normalized_mutual_weight(
                        graph,
                        nbr,
                        mid,
                        weight_fn=weight_fn,
                        norm=max,
                        neighbors=neighbors,
                    )
                )
            total += 1.0 - redundancy
        result[store.index_to_id[node_idx]] = total
    return result
```

### Bipartite Weighted Projection and Degree Centrality

Design:
- Use undirected neighbor sets for bipartite graphs.
- Match NetworkX `weighted_projected_graph`:
  - `weight = number of shared neighbors` by default.
  - `ratio=True` uses `weight = shared / size(other_partition)`.
- Degree centrality:
  - For nodes in U: `deg(v) / |V|`.
  - For nodes in V: `deg(v) / |U|`.

```python
def weighted_projected_graph(
    store: RxGraphStore,
    nodes: set[Hashable],
    *,
    ratio: bool = False,
) -> RxGraphStore:
    graph = store.graph
    node_indices = {store.id_to_index[n] for n in nodes if n in store.id_to_index}
    if len(node_indices) >= graph.num_nodes():
        message = "projection nodes must be a strict subset of graph nodes"
        raise ValueError(message)
    neighbors = _undirected_neighbors(graph)
    projected = RxGraphStore.build()
    for node_idx in node_indices:
        projected.ensure_node(store.index_to_id[node_idx])
    other_size = graph.num_nodes() - len(node_indices)
    for left in node_indices:
        left_nbrs = neighbors.get(left, set())
        candidates = {n for nbr in left_nbrs for n in neighbors.get(nbr, set())} - {left}
        for right in candidates:
            if right not in node_indices:
                continue
            common = left_nbrs & neighbors.get(right, set())
            if not common:
                continue
            weight = len(common) / float(other_size) if ratio else float(len(common))
            add_weighted_edge(
                projected,
                store.index_to_id[left],
                store.index_to_id[right],
            )
            edge_idx_left = projected.id_to_index[store.index_to_id[left]]
            edge_idx_right = projected.id_to_index[store.index_to_id[right]]
            projected.graph.update_edge(edge_idx_left, edge_idx_right, weight)
    return projected


def bipartite_degree_centrality(
    store: RxGraphStore,
    top_nodes: set[Hashable],
) -> dict[Hashable, float]:
    all_nodes = {store.index_to_id[idx] for idx in store.graph.node_indices()}
    bottom_nodes = all_nodes - top_nodes
    top_scale = 1.0 / float(len(bottom_nodes)) if bottom_nodes else 0.0
    bottom_scale = 1.0 / float(len(top_nodes)) if top_nodes else 0.0
    neighbors = _undirected_neighbors(store.graph)
    result: dict[Hashable, float] = {}
    for node_id in all_nodes:
        node_idx = store.id_to_index[node_id]
        degree = len(neighbors.get(node_idx, set()))
        scale = top_scale if node_id in top_nodes else bottom_scale
        result[node_id] = float(degree) * scale
    return result
```

## Deterministic Ordering, NaN Handling, and Numeric Tolerances

These policies apply to all custom algorithms to ensure stable outputs and
testable numeric behavior.

### Deterministic Ordering
- Always materialize results in a deterministic order, even when the return
  type is a dict. Build dicts from sorted keys to preserve insertion order.
- For sums of floats (clustering, constraint, effective size), iterate
  neighbors in sorted order to avoid non-deterministic accumulation.
- Use a stable key that tolerates mixed node types:

```python
def stable_key(value: object) -> tuple[str, str]:
    return (type(value).__name__, str(value))
```

### NaN Handling
- Harmonic centrality: isolated nodes or nodes with no reachable targets
  return `0.0` (matches NetworkX behavior).
- Clustering: nodes with degree < 2 return `0.0`.
- Triangles: nodes with degree < 2 return `0`.
- Constraint and effective size:
  - Isolated nodes (including self-loop-only) return `NaN` internally.
  - At analytics call sites, convert `NaN` to `0.0` to match existing
    behavior in `compute_constraint` and `compute_effective_size`.
- Bipartite projection:
  - Invalid inputs (empty nodes, nodes not in graph, or nodes >= graph size)
    return an empty graph rather than raising.

### Numeric Tolerance Policy

Use `math.isclose` with the following defaults when comparing against
NetworkX baselines or fixtures:

| Metric | abs_tol | rel_tol | Notes |
| --- | --- | --- | --- |
| Harmonic centrality | `1e-9` | `1e-6` | Path-length sums; floats accumulate. |
| Clustering (weighted) | `1e-9` | `1e-6` | Geometric mean of weights. |
| Clustering (unweighted) | `0.0` | `0.0` | Rational results; exact match expected. |
| Constraint | `1e-9` | `1e-6` | Sum of squared terms. |
| Effective size | `1e-9` | `1e-6` | Redundancy sums. |
| Projection weights (ratio) | `1e-12` | `1e-9` | Ratios of small integers. |
| Degree centrality (bipartite) | `1e-12` | `1e-9` | Exact ratios. |

For integer-valued outputs (triangles, component sizes, counts), require
exact equality.

## New Rustworkx Code Patterns

### A) Graph Store with Stable ID Mapping
```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable

import rustworkx as rx


@dataclass(slots=True)
class RxGraphStore:
    graph: rx.PyDiGraph
    id_to_index: dict[Hashable, int]
    index_to_id: dict[int, Hashable]

    @classmethod
    def build(cls, *, node_hint: int | None = None, edge_hint: int | None = None) -> "RxGraphStore":
        graph = rx.PyDiGraph(
            multigraph=False,
            node_count_hint=node_hint,
            edge_count_hint=edge_hint,
        )
        return cls(graph=graph, id_to_index={}, index_to_id={})

    def ensure_node(self, node_id: Hashable) -> int:
        existing = self.id_to_index.get(node_id)
        if existing is not None:
            return existing
        index = self.graph.add_node(node_id)
        self.id_to_index[node_id] = index
        self.index_to_id[index] = node_id
        return index
```

### B) Weighted Edge Aggregation (No Multi-Edges)
```python
def add_weighted_edge(store: RxGraphStore, src_id: Hashable, dst_id: Hashable) -> None:
    src_idx = store.ensure_node(src_id)
    dst_idx = store.ensure_node(dst_id)
    if store.graph.has_edge(src_idx, dst_idx):
        current = store.graph.get_edge_data(src_idx, dst_idx)
        weight = int(current or 0) + 1
        store.graph.update_edge(src_idx, dst_idx, weight)
        return
    store.graph.add_edge(src_idx, dst_idx, 1)
```

### C) Weight Function Policy
```python
def edge_weight(payload: object) -> float:
    if payload is None:
        return 1.0
    if isinstance(payload, bool):
        return float(int(payload))
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, str):
        try:
            return float(payload)
        except ValueError:
            return 1.0
    return 1.0
```

### D) Centrality with ID Mapping
```python
import rustworkx as rx


def pagerank_by_id(store: RxGraphStore) -> dict[int, float]:
    raw = rx.pagerank(store.graph, weight_fn=edge_weight, default_weight=1.0)
    return {store.index_to_id[idx]: float(score) for idx, score in raw.items()}
```

### E) Directed Simple Paths for Config Data Flow
```python
import rustworkx as rx


def all_simple_paths_by_id(
    store: RxGraphStore,
    sources: list[int],
    target: int,
    *,
    cutoff: int,
    limit: int,
) -> list[list[int]]:
    target_idx = store.id_to_index.get(target)
    if target_idx is None:
        return [[target]]
    results: list[list[int]] = []
    for source_id in sources:
        source_idx = store.id_to_index.get(source_id)
        if source_idx is None:
            continue
        for path in rx.digraph_all_simple_paths(store.graph, source_idx, target_idx, cutoff):
            results.append([store.index_to_id[idx] for idx in path])
            if len(results) >= limit:
                return results
    return results or [[target]]
```

### F) SCC Condensation (Custom)
```python
import rustworkx as rx


def condensation_graph(store: RxGraphStore) -> tuple[rx.PyDiGraph, dict[int, int]]:
    sccs = list(rx.strongly_connected_components(store.graph))
    comp_map: dict[int, int] = {}
    for comp_idx, comp in enumerate(sccs):
        for node_idx in comp:
            comp_map[node_idx] = comp_idx
    condensed = rx.PyDiGraph(multigraph=False)
    condensed.add_nodes_from(range(len(sccs)))
    for src, dst in store.graph.edge_list():
        src_comp = comp_map.get(src)
        dst_comp = comp_map.get(dst)
        if src_comp is None or dst_comp is None or src_comp == dst_comp:
            continue
        condensed.add_edge(src_comp, dst_comp, 1)
    return condensed, comp_map
```

### G) Serialization and Cache
```python
from pathlib import Path
import rustworkx as rx


def write_graph(path: Path, graph: rx.PyDiGraph) -> None:
    payload = rx.node_link_json(graph)
    path.write_text(payload, encoding="utf-8")


def read_graph(path: Path) -> rx.PyDiGraph:
    payload = path.read_text(encoding="utf-8")
    return rx.parse_node_link_json(payload)
```

## Testing and Validation Strategy
- Update fixture factory: `tests/_helpers/fixtures/graphs.py`
  - Replace `nx.Graph()` / `nx.DiGraph()` with rustworkx stores.
  - Add deterministic conversion helpers for expected outputs.
- Update graph algorithm tests:
  - Map node indices to domain IDs before assertions.
  - Use stable ordering on rustworkx return types.
- Run focused tests per phase:
  - Graph builders and engine tests under `tests/graphs/*`.
  - Analytics compute tests under `tests/analytics/compute/*`.

## Definition of Done
- No imports of `networkx` remain under `src/` or `tests/`.
- `pyproject.toml` and `uv.lock` contain rustworkx and no networkx packages.
- Graph metrics tables match existing schemas and row counts.
- All tests pass and new rustworkx patterns are in place.
