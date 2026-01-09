# Rustworkx Best-in-Class Alignment Plan

## Objective
Consolidate rustworkx usage into a minimal, reusable surface that maximizes
library capabilities, reduces bespoke code, and strengthens determinism,
extensibility, and maintainability across graph analytics.

## Design Principles (Non-Negotiable)
- Algorithm calls are centralized and type-specific (`digraph_*`/`graph_*`).
- Weight semantics are explicit (STRENGTH vs COST) at every weighted call site.
- Graph construction uses bulk APIs with capacity hints and stable ordering.
- Graph views and metrics use rustworkx primitives (subgraph/union/condensation).
- Serialization and metadata are deterministic and schema-aligned.

---

## Scope 01 - Typed Algorithm Envelope (Single Entry Point)
**Goal**
Route all rustworkx algorithm calls through a single typed envelope so directed
vs undirected semantics, weight semantics, and output normalization are enforced
uniformly.

**Code patterns**
```python
from codeintel.build.graphs.rx.algos import (
    GraphAlgoConfig,
    GraphInput,
    digraph_shortest_path_lengths,
)

config = GraphAlgoConfig(weight_semantics="cost", rayon_threads=8)
lengths = digraph_shortest_path_lengths(
    graph,
    source_id=source_id,
    config=config,
)
```

```python
from codeintel.build.graphs.rx.algos import (
    graph_betweenness_centrality,
    normalize_mapping,
)

raw = graph_betweenness_centrality(graph, config=config)
normalized = normalize_mapping(raw, nan_policy="zero")
```

**Target files**
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/compute/metrics/components.py`
- `src/codeintel/build/graphs/compute/metrics/dfg.py`
- `src/codeintel/build/graphs/compute/metrics/paths.py`
- `src/codeintel/build/graphs/compute/metrics/statistics.py`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/graphs/validation/checks/structure.py`

**Implementation checklist**
- [ ] Add typed wrappers for all algorithm families used by metrics modules.
- [ ] Replace direct `rx.*` calls in metrics/analytics with typed wrappers.
- [ ] Normalize outputs (sorted mapping, stable ordering) in the envelope only.
- [ ] Centralize weight function construction in `rx.algos`.

---

## Scope 02 - Iterator and Return-Type Normalization
**Goal**
Replace bespoke edge/node iteration with rustworkx return types and shared
iterators to ensure consistent ordering and payload handling.

**Code patterns**
```python
from codeintel.build.graphs.rx.iterators import iter_edge_payloads

for src_idx, dst_idx, payload in iter_edge_payloads(store):
    weight = weight_policy.normalize_weight(payload)
```

```python
from codeintel.build.graphs.rx.normalize import sorted_mapping

centrality = sorted_mapping(raw_centrality)
```

**Target files**
- `src/codeintel/build/graphs/rx/iterators.py`
- `src/codeintel/build/graphs/rx/normalize.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`
- `src/codeintel/build/analytics/graphs/subsystem_graph_metrics.py`
- `src/codeintel/build/analytics/subsystems/affinity.py`
- `src/codeintel/build/graphs/compute/metrics/community.py`
- `src/codeintel/build/graphs/compute/metrics/cfg.py`

**Implementation checklist**
- [ ] Add iterators for edge_index maps, weighted edge lists, and neighbors.
- [ ] Replace manual edge_list loops in analytics with iterators.
- [ ] Enforce stable ordering through `stable_key` and `sorted_mapping`.

---

## Scope 03 - Graph Operations Over Bespoke Subgraph/Merge Logic
**Goal**
Use rustworkx primitives for subgraphs, merges, and condensation so graph
composition is correct and fast.

**Code patterns**
```python
import rustworkx as rx

subgraph, node_map = rx.subgraph_with_nodemap(graph, node_indices)
merged = rx.union(graph, other_graph, merge_nodes=True, merge_edges=True)
```

```python
condensed = rx.condensation(digraph)
node_map = condensed.attrs.get("node_map")
```

**Target files**
- `src/codeintel/build/graphs/compute/metrics/components.py`
- `src/codeintel/build/graphs/compute/imports.py`
- `src/codeintel/build/graphs/compute/metrics/statistics.py`
- `src/codeintel/build/graphs/rx/condensation.py`

**Implementation checklist**
- [ ] Replace manual condensation rebuilds with `rx.condensation` + node_map.
- [ ] Use `rx.subgraph_with_nodemap` for filtered graph views.
- [ ] Use `rx.union`/`rx.compose` for merge operations.

---

## Scope 04 - Best-in-Class Graph Construction
**Goal**
Ensure graph building uses rustworkx bulk APIs with deterministic ordering and
capacity hints for high-volume pipelines.

**Code patterns**
```python
store = RxGraphStore.directed(node_hint=100_000, edge_hint=500_000)
store.graph.add_nodes_from(node_payloads)
store.graph.add_edges_from(edge_triples)
```

```python
from codeintel.build.graphs.rx.build_from_edges import BuildStoreOptions

store = build_store_from_edge_tuples(
    edges,
    spec=spec,
    options=BuildStoreOptions(stable_nodes=True, aggregate_edges=True),
)
```

**Target files**
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/rx/store.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/graphs/engine/views.py`

**Implementation checklist**
- [ ] Ensure node/edge count hints are propagated at construction sites.
- [ ] Prefer bulk add methods (`add_nodes_from`, `add_edges_from`).
- [ ] Centralize aggregate edge behavior (no per-module loops).

---

## Scope 05 - Serialization and Graph Metadata
**Goal**
Make serialization deterministic and round-trip safe, and standardize graph
metadata via rustworkx `attrs`.

**Code patterns**
```python
payload = rx.digraph_node_link_json(
    graph,
    graph_attrs=graph_attrs_in,
    node_attrs=pack_payload,
    edge_attrs=pack_payload,
)
graph = rx.parse_node_link_json(payload, graph_attrs=graph_attrs_out)
```

```python
apply_graph_metadata(
    graph,
    GraphMetadata(weight_policy="strength", determinism_tier="canonical"),
)
```

**Target files**
- `src/codeintel/build/graphs/rx/serialization.py`
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/rx/store.py`

**Implementation checklist**
- [ ] Use type-specific node-link JSON functions when directed/undirected known.
- [ ] Persist determinism tier, ordering keys, and weight policy in `graph.attrs`.
- [ ] Enforce metadata presence for cached serialization flows.

---

## Scope 06 - Traversal and Path APIs
**Goal**
Replace bespoke traversal/path loops with rustworkx traversal APIs and visitors
to reduce code size and improve correctness.

**Code patterns**
```python
from rustworkx import visit

visitor = visit.BFSVisitor()
rx.digraph_bfs_search(digraph, [source_idx], visitor)
```

```python
lengths = rx.digraph_dijkstra_shortest_path_lengths(digraph, source_idx, weight_fn)
```

**Target files**
- `src/codeintel/build/graphs/compute/metrics/dfg.py`
- `src/codeintel/build/graphs/compute/metrics/paths.py`
- `src/codeintel/build/graphs/compute/metrics/statistics.py`

**Implementation checklist**
- [ ] Use BFS/DFS visitor hooks for bounded traversals.
- [ ] Use shortest path APIs for reachability metrics.
- [ ] Centralize traversal patterns inside `rx.algos`.

---

## Scope 07 - Weight Semantics Everywhere
**Goal**
Ensure all weighted algorithm calls explicitly declare strength/cost semantics
and normalize edge payloads consistently.

**Code patterns**
```python
from codeintel.build.graphs.rx.algos import (
    resolve_weight_context,
    edge_cost_weight_fn,
)

context = resolve_weight_context(store, algo_config=config)
weight_fn = edge_cost_weight_fn(context=context)
lengths = rx.digraph_dijkstra_shortest_path_lengths(graph, source_idx, weight_fn)
```

**Target files**
- `src/codeintel/build/graphs/rx/weights.py`
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/compute/metrics/*`
- `src/codeintel/build/analytics/graphs/*`

**Implementation checklist**
- [ ] Replace raw payload usage with semantics-aware weight helpers.
- [ ] Enforce NaN policies for all numeric outputs.
- [ ] Add centralized defaults for weight epsilon and fallback behavior.

---

## Sequencing Recommendation
1) Scope 01 - Typed algorithm envelope
2) Scope 02 - Iterator and return-type normalization
3) Scope 03 - Graph operations replacement
4) Scope 04 - Graph construction optimization
5) Scope 05 - Serialization + metadata
6) Scope 06 - Traversal/path APIs
7) Scope 07 - Weight semantics enforcement

## Validation Gates (Guardrails Deferred)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted pytest subsets for `build/graphs` and analytics graph modules once tests resume
