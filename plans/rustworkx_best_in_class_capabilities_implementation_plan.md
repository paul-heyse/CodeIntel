# Rustworkx Best-in-Class Capabilities Implementation Plan

## Objective
Create a unified, high-performance rustworkx surface that minimizes bespoke graph logic,
keeps Arrow Acero/DSL as the compute lane for table assembly, and makes graph construction
the only rustworkx boundary. Every graph operation should be typed, deterministic, and
explicit about weight semantics.

## Scope 1: Arrow-first graph assembly + finalize boundaries
**Goal:** construct edge/node tables using Acero/DSL and finalize them (ordering + contracts)
before rustworkx ingestion.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.plan_ops import Plan

plan = (
    Plan.table(edge_table)
    .project(
        src=E.field("caller_goid_h128"),
        dst=E.field("callee_goid_h128"),
    )
    .hash_join(
        right=module_map,
        key_pairs=[("dst", "goid_h128")],
    )
    .aggregate(keys=["src", "dst"], aggregates=[("src", "count", "weight")])
)

edges = ExecutionPlan.from_plan(plan, determinism="canonical").to_table(ctx=context)
edges = finalize_table(
    edges,
    FinalizeSpec(
        name="graph_edges",
        ordering_keys=["src", "dst"],
        determinism="canonical",
    ),
)
```

**Target files**
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`
- `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/finalize_ops.py`

**Implementation checklist**
- [ ] Replace row-wise graph assembly with Acero/DSL plans for all graph loaders.
- [ ] Convert row-based call/import/symbol graph loaders to table-first plans.
- [ ] Add finalize boundaries right before rustworkx ingestion.
- [ ] Enforce canonical ordering using contract keys (plus provenance tie-breakers).

**Remaining scope focus**
- Migrate row-driven graph builders in `src/codeintel/build/graphs/builders.py`.
- Move analytics graph metrics loaders to Acero/DSL + finalize in
  `src/codeintel/build/analytics/graphs/graph_metrics.py` and
  `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`.
- Align Hamilton analytics graph metrics with table-first plans in
  `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`.

---

## Scope 2: Unified GraphBuilder + EdgeBuildSpec ingestion
**Goal:** all graph loaders route through a single EdgeBuildSpec-based builder with stable
node ordering and bulk edge ingestion.

**Code pattern**
```python
from codeintel.build.graphs.rx.build_from_edges import BuildStoreOptions, EdgeBuildSpec
from codeintel.build.graphs.rx.build_from_edges import build_store_from_edge_tuples
from codeintel.build.graphs.rx.policies import DEFAULT_NUMERIC_POLICY, weight_policy_for_kind
from codeintel.build.graphs.rx.store import GraphKind

spec = EdgeBuildSpec(
    directed=True,
    weight_policy=weight_policy_for_kind(GraphKind.CALL_GRAPH),
    numeric_policy=DEFAULT_NUMERIC_POLICY,
)
options = BuildStoreOptions(node_hint=200_000, edge_hint=2_000_000, stable_nodes=True)

store = build_store_from_edge_tuples(edge_rows, spec=spec, options=options)
```

**Target files**
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`

**Implementation checklist**
- [ ] Consolidate all graph loaders into `build_store_from_edge_tuples`.
- [ ] Supply stable node lists and capacity hints for large graphs.
- [ ] Remove bespoke per-edge `add_edge` loops in loaders and metrics.

**Remaining scope focus**
- Retire row-based helpers (`build_*_from_rows`, `add_*_edges`) once all callers
  are table-first.
- Ensure graph loaders pass `node_ids`/`node_attrs` explicitly rather than
  inferring via per-edge loops.

---

## Scope 3: Ordering + determinism metadata propagation
**Goal:** ordering intent is explicit at the plan level, and finalize uses it for canonical
determinism; store metadata captures determinism + ordering keys.

**Code pattern**
```python
from codeintel.core.columnar.ordering import OrderingSpec
from codeintel.core.columnar.plan_ops import Plan

ordering = OrderingSpec(keys=[("src", "ascending"), ("dst", "ascending")])
plan = Plan.table(edge_table).order_by(ordering)
```

**Target files**
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/build/graphs/assembly/finalize.py`
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/runtime/runtime.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`

**Implementation checklist**
- [ ] Propagate OrderingSpec through plan nodes and into finalize.
- [ ] Enforce canonical ordering at finalize boundaries (no ad hoc sorting).
- [ ] Persist ordering keys and determinism tier in graph metadata.

**Remaining scope focus**
- Attach ordering metadata for graphs built outside `views.py` so cached
  graphs carry determinism + ordering keys.
- Route ordering keys from contract schemas into finalize for row-based
  loaders that still exist.

---

## Scope 4: Typed algorithm envelope + weight semantics
**Goal:** all rustworkx algorithms go through typed wrappers with explicit weight semantics
and normalized output ordering.

**Code pattern**
```python
from codeintel.build.graphs.rx.algos import GraphAlgoConfig
from codeintel.build.graphs.rx.algos import edge_cost_weight_fn, resolve_weight_context
from codeintel.build.graphs.rx.algos import digraph_katz_centrality_by_id

config = GraphAlgoConfig(weight_semantics="cost")
context = resolve_weight_context(store, algo_config=config)
weight_fn = edge_cost_weight_fn(context=context)
scores = digraph_katz_centrality_by_id(store, weight_fn=weight_fn, algo_config=config)
```

```python
from codeintel.build.graphs.rx.algos import bfs_distances_by_id

distances = bfs_distances_by_id(call_graph_store, source_id, max_depth=5)
```

**Target files**
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/rx/weights.py`
- `src/codeintel/build/graphs/compute/metrics/*`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/analytics/functions/function_effects.py`

**Implementation checklist**
- [ ] Add wrappers for HITS, Katz, transitivity, edge betweenness, and additional shortest paths.
- [ ] Replace direct rustworkx traversal logic with typed wrappers.
- [ ] Route weighted algorithms through `resolve_weight_context`.
- [ ] Normalize outputs with stable ordering and NaN policy.

**Remaining scope focus**
- Implement missing wrappers in `src/codeintel/build/graphs/rx/algos.py`
  (HITS/Katz/transitivity/edge betweenness/Floyd-Warshall/Bellman-Ford/k-shortest).
- Replace the BFS traversal in
  `src/codeintel/build/analytics/functions/function_effects.py` with the
  `bfs_distances_by_id` wrapper.

---

## Scope 5: Rustworkx primitives for components, subgraphs, and merges
**Goal:** replace bespoke graph logic with built-in rustworkx primitives.

**Code pattern**
```python
import rustworkx as rx

condensed = rx.condensation(store.graph)
subgraph, node_map = rx.subgraph_with_nodemap(store.graph, node_indices, preserve_attrs=True)
layers = rx.layers(condensed)
merged = rx.union(graph_a, graph_b, merge_nodes=True, merge_edges=True)
```

**Target files**
- `src/codeintel/build/graphs/compute/metrics/components.py`
- `src/codeintel/build/graphs/compute/metrics/cfg.py`
- `src/codeintel/build/graphs/compute/metrics/statistics.py`
- `src/codeintel/build/graphs/compute/metrics/community.py`
- `src/codeintel/build/graphs/compute/imports.py`

**Implementation checklist**
- [ ] Replace SCC/condensation logic with `rx.condensation`.
- [ ] Use `subgraph_with_nodemap` for filtered views.
- [ ] Use `rx.layers` / `rx.topological_generations` for DAG layers.
- [ ] Use `rx.union` / `rx.compose` for graph merges.

**Remaining scope focus**
- Reduce bespoke merge/subgraph logic in `src/codeintel/build/graphs/compute/metrics/community.py`
  by routing through shared rustworkx primitives.

---

## Scope 6: Return-type normalization + iterators
**Goal:** use shared iterators and normalize rustworkx return types centrally.

**Code pattern**
```python
from codeintel.build.graphs.rx.iterators import iter_edge_id_weights
from codeintel.build.graphs.rx.normalize import sorted_mapping

weights = {src: 0.0 for src, _, _ in iter_edge_id_weights(store)}
weights = sorted_mapping(weights)
```

**Target files**
- `src/codeintel/build/graphs/rx/iterators.py`
- `src/codeintel/build/graphs/rx/normalize.py`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`

**Implementation checklist**
- [ ] Add missing iterators (edge index map, payload tuples, incident edges).
- [ ] Replace manual edge_list loops with iterators.
- [ ] Normalize nested mappings via shared helpers.

**Remaining scope focus**
- Add an incident-edges iterator (or equivalent) to `src/codeintel/build/graphs/rx/iterators.py`
  if any remaining algorithms need direct edge adjacency.
- Remove any residual direct rustworkx traversal loops in analytics helpers.

---

## Scope 7: Serialization + metadata (node-link JSON)
**Goal:** deterministic, lossless serialization with explicit metadata.

**Code pattern**
```python
from codeintel.build.graphs.rx.metadata import GraphMetadata, apply_graph_metadata
from codeintel.build.graphs.rx.serialization import dumps_node_link_json

apply_graph_metadata(
    store.graph,
    GraphMetadata(weight_policy="strength", determinism_tier="canonical"),
)
payload = dumps_node_link_json(store.graph, require_metadata=True)
```

**Target files**
- `src/codeintel/build/graphs/rx/serialization.py`
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/runtime/runtime.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`
- `src/codeintel/build/hamilton/native/analytics/graph_metrics.py`

**Implementation checklist**
- [ ] Use typed node-link JSON for both directed and undirected graphs.
- [ ] Persist determinism tier + ordering keys + weight policy in attrs.
- [ ] Ensure round-trip preserves node IDs and payloads.

**Remaining scope focus**
- Apply `GraphMetadata` in row-based graph builders so cached graphs
  include determinism tier and ordering keys.

---

## Scope 8: Directed mutation helpers + transform utilities
**Goal:** use rustworkx mutation helpers for CFG/DFG transforms instead of
manual rebuilds.

**Code pattern**
```python
from codeintel.build.graphs.rx.algos import (
    insert_node_on_out_edges_by_id,
    remove_node_retain_edges_by_id,
)

insert_node_on_out_edges_by_id(store, new_node_id, ref_node_id, attrs={"kind": "phi"})
remove_node_retain_edges_by_id(store, obsolete_node_id, use_outgoing=True)
```

**Target files**
- `src/codeintel/build/graphs/compute/metrics/cfg.py`
- `src/codeintel/build/graphs/compute/metrics/dfg.py`
- `src/codeintel/build/analytics/cfg_dfg/*`

**Implementation checklist**
- [ ] Replace manual node insertion/removal logic with mutation helpers.
- [ ] Centralize transform helpers in `rx.algos` or `rx.transforms`.

**Remaining scope focus**
- Adopt the directed mutation helpers in CFG/DFG transforms in
  `src/codeintel/build/graphs/compute/metrics/cfg.py` and
  `src/codeintel/build/graphs/compute/metrics/dfg.py`.

---

## Scope 9: Construction performance + capacity hints
**Goal:** preallocate and bulk-insert edges everywhere for throughput.

**Code pattern**
```python
from codeintel.build.graphs.rx.build_from_edges import BuildStoreOptions

options = BuildStoreOptions(node_hint=500_000, edge_hint=5_000_000, stable_nodes=True)
store = build_store_from_edge_tuples(edge_rows, spec=spec, options=options)
```

**Target files**
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`

**Implementation checklist**
- [ ] Always pass capacity hints for large graphs.
- [ ] Prefer bulk edge insertion; avoid per-edge Python loops.
- [ ] Use aggregated edge rows (post-DSL) to reduce insert volume.

**Remaining scope focus**
- Replace row-filtered graph rebuilds with bulk ingestion or subgraph primitives
  in analytics graph metrics and symbol graph metrics loaders.

---

## Suggested execution order
1) Scope 1: Arrow-first graph assembly + finalize boundaries
2) Scope 2: Unified GraphBuilder + EdgeBuildSpec ingestion
3) Scope 3: Ordering + determinism metadata propagation
4) Scope 4: Typed algorithm envelope + weight semantics
5) Scope 5: Rustworkx primitives for components/subgraphs/merges
6) Scope 6: Return-type normalization + iterators
7) Scope 7: Serialization + metadata
8) Scope 8: Directed mutation helpers
9) Scope 9: Construction performance + capacity hints

## Validation (when tests resume)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted graph analytics and metrics tests for affected modules
