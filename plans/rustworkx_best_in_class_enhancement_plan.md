# Rustworkx Best-in-Class Enhancement Plan

## Objective
Expand and unify rustworkx usage so graph analytics rely on a centralized,
typed API surface with deterministic outputs, explicit weight semantics,
and minimal bespoke graph logic. This plan complements the Arrow Acero/DSL
pipeline by keeping final graph construction as the only rustworkx boundary.

## Guiding Principles
- All algorithm calls go through typed wrappers (digraph_* / graph_*).
- Weight semantics (strength vs cost) are explicit at every weighted call.
- Bulk construction + capacity hints are default for graph ingestion.
- Outputs are deterministic (stable ordering, normalized mappings).
- Graph metadata captures determinism tier + ordering keys + weight policy.

---

## Scope 1 — Typed Algorithm Envelope Expansion
**Goal:** cover the full rustworkx algorithm surface used by analytics and
validation, with normalized, typed outputs and consistent semantics.

**Code pattern**
```python
from codeintel.build.graphs.rx.algos import (
    GraphAlgoConfig,
    digraph_katz_centrality_by_id,
    edge_cost_weight_fn,
    resolve_weight_context,
)

config = GraphAlgoConfig(weight_semantics="cost", rayon_threads=8)
context = resolve_weight_context(store, algo_config=config)
weight_fn = edge_cost_weight_fn(context=context)
scores = digraph_katz_centrality_by_id(store, weight_fn=weight_fn, algo_config=config)
```

**Target files**
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/compute/metrics/*`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/graphs/validation/checks/structure.py`

**Implementation checklist**
- [ ] Add wrappers for hits/katz, edge betweenness, transitivity, and graph measures.
- [ ] Add typed shortest-path wrappers for Bellman-Ford/Floyd-Warshall/k-shortest paths.
- [ ] Add matching/coloring wrappers for analytics or validation use.
- [ ] Normalize outputs with stable ordering + NaN policy in the envelope.

---

## Scope 2 — Traversal and Visitor Unification
**Goal:** replace bespoke traversal loops with rustworkx BFS/DFS/Dijkstra visitors.

**Code pattern**
```python
from codeintel.build.graphs.rx.algos import bfs_distances_by_id

distances = bfs_distances_by_id(store, source_id, max_depth=32)
```

**Target files**
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/compute/metrics/dfg.py`
- `src/codeintel/build/analytics/functions/function_effects.py`
- `src/codeintel/build/analytics/cfg_dfg/dfg_core.py`

**Implementation checklist**
- [ ] Centralize BFS/DFS/Dijkstra visitors in `rx.algos`.
- [ ] Replace custom BFS loops with visitor wrappers.
- [ ] Ensure deterministic ordering of traversal outputs.

---

## Scope 3 — Return-Type Normalization and Iterators
**Goal:** rely on rustworkx structured return types and shared iterators, and
normalize outputs in one place.

**Code pattern**
```python
from codeintel.build.graphs.rx.iterators import iter_edge_id_weights
from codeintel.build.graphs.rx.normalize import sorted_mapping

counts = {src: counts.get(src, 0.0) + weight for src, _, weight in iter_edge_id_weights(store)}
counts = sorted_mapping(counts)
```

**Target files**
- `src/codeintel/build/graphs/rx/iterators.py`
- `src/codeintel/build/graphs/rx/normalize.py`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`

**Implementation checklist**
- [ ] Add missing iterators (edge_index_map, weighted lists, incident edges).
- [ ] Replace manual edge_list loops with iterators.
- [ ] Normalize nested mappings using shared helpers.

---

## Scope 4 — Graph Ops and Composition
**Goal:** use rustworkx composition, subgraph, and condensation primitives
instead of bespoke merge/subgraph logic.

**Code pattern**
```python
import rustworkx as rx

subgraph, node_map = rx.subgraph_with_nodemap(graph, node_indices, preserve_attrs=True)
merged = rx.union(graph, other, merge_nodes=True, merge_edges=True)
```

**Target files**
- `src/codeintel/build/graphs/compute/metrics/community.py`
- `src/codeintel/build/graphs/compute/metrics/components.py`
- `src/codeintel/build/graphs/compute/imports.py`
- `src/codeintel/build/graphs/validation/checks/structure.py`

**Implementation checklist**
- [ ] Replace bespoke subgraph/merge logic with rustworkx ops.
- [ ] Use condensation with node_map where SCC mapping is required.
- [ ] Preserve node/edge payloads and stable ordering in subgraphs.

---

## Scope 5 — Directed Mutation Helpers
**Goal:** leverage PyDiGraph mutation helpers for CFG/DFG transforms and
structural edits.

**Code pattern**
```python
directed.insert_node_on_out_edges(node_idx, new_payload)
directed.remove_node_retain_edges(node_idx)
```

**Target files**
- `src/codeintel/build/graphs/compute/metrics/cfg.py`
- `src/codeintel/build/graphs/compute/metrics/dfg.py`
- `src/codeintel/build/analytics/cfg_dfg/*`

**Implementation checklist**
- [ ] Replace manual rebuilds with insert/remove retain-edge helpers.
- [ ] Centralize transform helpers in `rx.algos` or a new `rx.transforms`.

---

## Scope 6 — Weight Semantics Everywhere
**Goal:** enforce explicit cost/strength semantics for all weighted algorithms.

**Code pattern**
```python
from codeintel.build.graphs.rx.algos import GraphAlgoConfig, resolve_weight_context
from codeintel.build.graphs.rx.algos import edge_cost_weight_fn

config = GraphAlgoConfig(weight_semantics="cost")
context = resolve_weight_context(store, algo_config=config)
weight_fn = edge_cost_weight_fn(context=context)
```

**Target files**
- `src/codeintel/build/graphs/rx/weights.py`
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/compute/metrics/*`
- `src/codeintel/build/analytics/graphs/*`

**Implementation checklist**
- [ ] Route all weighted calls through `resolve_weight_context`.
- [ ] Enforce NaN policies + epsilon handling at wrapper boundary.
- [ ] Remove direct payload-to-weight conversions at call sites.

---

## Scope 7 — Serialization + Metadata
**Goal:** deterministic, lossless serialization with explicit graph metadata.

**Code pattern**
```python
from codeintel.build.graphs.rx.serialization import dumps_node_link_json
from codeintel.build.graphs.rx.metadata import GraphMetadata, apply_graph_metadata

apply_graph_metadata(graph, GraphMetadata(weight_policy="strength", determinism_tier="canonical"))
payload = dumps_node_link_json(graph, require_metadata=True)
```

**Target files**
- `src/codeintel/build/graphs/rx/serialization.py`
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/runtime/runtime.py`

**Implementation checklist**
- [ ] Use typed node-link JSON functions for directed/undirected graphs.
- [ ] Persist determinism tier + ordering keys + weight policy in attrs.
- [ ] Add GraphML import/export helpers where needed.

---

## Scope 8 — Construction Performance + Capacity Hints
**Goal:** bulk ingestion with preallocation and stable ordering by default.

**Code pattern**
```python
from codeintel.build.graphs.rx.build_from_edges import BuildStoreOptions

options = BuildStoreOptions(node_hint=100_000, edge_hint=500_000, stable_nodes=True)
store = build_store_from_edge_tuples(edges, spec=spec, options=options)
```

**Target files**
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/graphs/compute/imports.py`
- `src/codeintel/build/graphs/engine/views.py`

**Implementation checklist**
- [ ] Always pass node/edge hints for large builders.
- [ ] Use bulk edge insertion for row-driven builds.
- [ ] Avoid per-edge Python loops where bulk APIs exist.

---

## Execution Order (Recommended)
1) Scope 1 — Typed algorithm envelope expansion
2) Scope 2 — Traversal/visitor unification
3) Scope 3 — Return-type normalization + iterators
4) Scope 4 — Graph ops and composition
5) Scope 6 — Weight semantics enforcement
6) Scope 7 — Serialization + metadata
7) Scope 8 — Construction performance
8) Scope 5 — Directed mutation helpers

## Validation (when tests resume)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted graph metric subsets (once tests are unpaused)
