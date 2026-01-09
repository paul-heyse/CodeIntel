# Rustworkx Best-in-Class Implementation Plan

## Scope 1: Arrow-first graph assembly via Acero DSL (Plan lane + kernel lane)

### Goal
Move graph assembly and joins into Acero plans, keeping only explode/row-expansion and contract enforcement in kernel/finalize helpers. Eliminate Python row loops for graph assembly.

### Code patterns

```python
import pyarrow.acero as acero
import pyarrow.compute as pc

scan = acero.Declaration(
    "scan",
    acero.ScanNodeOptions(
        dataset,
        columns=["caller_goid_h128", "callee_goid_h128"],
        filter=pc.is_valid(pc.field("caller_goid_h128")) & pc.is_valid(pc.field("callee_goid_h128")),
        implicit_ordering=True,
        require_sequenced_output=False,
    ),
)

project = acero.Declaration(
    "project",
    acero.ProjectNodeOptions(
        expressions=[pc.field("caller_goid_h128"), pc.field("callee_goid_h128")],
        names=["src", "dst"],
    ),
    inputs=[scan],
)

agg = acero.Declaration(
    "aggregate",
    acero.AggregateNodeOptions(
        keys=[pc.field("src"), pc.field("dst")],
        aggregates=[(pc.field("src"), "count", None, "weight")],
    ),
    inputs=[project],
)

reader = agg.to_reader(use_threads=True)
```

```python
import pyarrow.compute as pc

# Deterministic canonicalization before graph build when required
sort_keys = [("src", "ascending"), ("dst", "ascending")]
idx = pc.sort_indices(edge_table, sort_keys=sort_keys)
edge_table = edge_table.take(idx)
```

### Target files
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`
- `src/codeintel/build/analytics/cfg_dfg/cfg_core.py`
- `src/codeintel/ingestion/compute/dis_extract.py`
- `src/codeintel/ingestion/compute/inspect_extract.py`

### Implementation checklist
- [x] Add Acero plan specs per graph type (call/import graph assembly via `Plan.table`).
- [x] Use `hashjoin` for mapping enrichment (no Python join loops).
- [x] Replace row-wise edge loops with `table_to_reader()` and `build_store_from_edge_tuples`.
- [x] Add canonical sort gates before graph build when determinism requires it.
- [ ] Ensure finalize gates enforce contract + ordering for canonical tier.

### Status (completed in Scope 1)
- Call/import graph assembly now uses Acero `Plan` in `src/codeintel/build/graphs/builders.py`.
- Call/import analytics loaders now pass scoped Arrow tables (no row lists).
- Function effects call graph construction now uses the table-first builder.
- Views-based graph loaders now aggregate edges via `Plan.table(...).aggregate(...)`.
- Symbol module coupling now uses `HashJoinSpec` against module lookup tables.
- Config bipartite loader now emits edge tuples and builds via `EdgeBuildSpec`.

---

## Scope 2: Unified rustworkx GraphBuilder (bulk add + capacity hints)

### Goal
Use one builder to ingest aggregated edge tables with stable node lists and rustworkx capacity hints.

### Code patterns

```python
from codeintel.build.graphs.rx.build_from_edges import EdgeBuildSpec, build_store_from_edge_tuples
from codeintel.build.graphs.rx.policies import DEFAULT_NUMERIC_POLICY, weight_policy_for_kind

spec = EdgeBuildSpec(
    directed=True,
    weight_policy=weight_policy_for_kind(GraphKind.CALL_GRAPH),
    numeric_policy=DEFAULT_NUMERIC_POLICY,
    src_fn=normalize_decimal,
    dst_fn=normalize_decimal,
)

store = build_store_from_edge_tuples(
    iter_edge_tuples,
    spec=spec,
    stable_nodes=True,
    aggregate_edges=True,
    node_ids=node_ids,
    node_hint=len(node_ids),
    edge_hint=edge_table.num_rows,
)
```

### Target files
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`

### Implementation checklist
- [x] Require core loaders/builders to route through EdgeBuildSpec (call/import/symbol graphs).
- [x] Add stable node lists (sorted by `stable_key`) before bulk ingest.
- [x] Apply `node_count_hint` and `edge_count_hint` consistently.
- [x] Remove bespoke `add_edge` loops in remaining loaders/builders.

### Status (completed in Scope 2)
- Call/import/symbol graph builders now use EdgeBuildSpec + BuildStoreOptions.
- Graph metrics/subsystem loaders now rely on table-first builders (no row-edge loops).
- Config graph analytics now uses table-first call-graph builders.

---

## Scope 3: Replace bespoke graph transforms with rustworkx primitives

### Goal
Use built-in rustworkx algorithms for condensation, layer derivation, subgraph views, and merges.

### Code patterns

```python
import rustworkx as rx

condensed = rx.condensation(store.graph)
node_map = condensed.node_map
```

```python
# Topological layers for DAG after condensation
layers = rx.layers(condensed)
```

```python
# Filtered views without rebuilding graphs
subgraph, nodemap = store.graph.subgraph_with_nodemap(node_indices)
```

```python
# Overlay graphs via compose/union
combined = rx.union(graph_a, graph_b)
```

### Target files
- `src/codeintel/build/graphs/compute/metrics/cfg.py`
- `src/codeintel/build/graphs/compute/metrics/components.py`
- `src/codeintel/build/graphs/compute/imports.py`
- `src/codeintel/build/graphs/compute/metrics/statistics.py`
- `src/codeintel/build/graphs/compute/metrics/community.py`

### Implementation checklist
- [x] Replace custom SCC/condensation logic with `rx.condensation`.
- [x] Use `layers` or `topological_generations` for DAG layer outputs.
- [x] Use `subgraph_with_nodemap` for filtered metric graphs.
- [x] Replace manual merges with `union`/`compose`.

### Status (completed in Scope 3)
- CFG longest-path and statistics now run on `rx.condensation(...)` outputs.
- Import SCC/layer computation now uses rustworkx condensation + generations.
- Community bridge-split uses `rx.connected_components` over unioned graphs.

---

## Scope 4: Typed rustworkx algorithm envelope + weight semantics

### Goal
Use type-specific rustworkx APIs and enforce weight semantics (STRENGTH vs COST) centrally.

### Code patterns

```python
import rustworkx as rx

# Typed API for directed graph shortest paths
lengths = rx.digraph_dijkstra_shortest_path_lengths(
    digraph,
    source_idx,
    weight_fn,
)
```

```python
from codeintel.build.graphs.rx.weights import WeightSemantics

algo_config = GraphAlgoConfig(weight_semantics=WeightSemantics.STRENGTH)
```

### Target files
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/rx/weights.py`
- `src/codeintel/build/graphs/compute/metrics/paths.py`
- `src/codeintel/build/graphs/compute/metrics/centrality.py`
- `src/codeintel/build/graphs/compute/metrics/projections.py`

### Implementation checklist
- [x] Prefer `digraph_*` and `graph_*` functions over universal dispatch.
- [x] Route all weighted algorithms through `WeightSemantics`.
- [x] Normalize output mappings deterministically with `stable_key`.
- [x] Centralize conversion strength<->cost for shortest-path algorithms.

### Status (completed in Scope 4)
- Algorithm wrappers now resolve weight semantics/epsilon centrally.
- Path and projection metrics use centralized weight conversion helpers.
- Centrality outputs normalize deterministically via `stable_key`.

---

## Scope 5: Correct node-link JSON + graph attrs metadata

### Goal
Guarantee lossless rustworkx serialization and track cache metadata in graph attrs.

### Code patterns

```python
import json
import rustworkx as rx

serialized = rx.node_link_json(
    graph,
    graph_attrs=lambda attrs: {str(k): str(v) for k, v in (attrs or {}).items()},
    node_attrs=lambda payload: {"payload": json.dumps(payload, separators=(",", ":"), sort_keys=True)},
    edge_attrs=lambda payload: {"payload": json.dumps(payload, separators=(",", ":"), sort_keys=True)},
)
```

```python
graph = rx.parse_node_link_json(
    data,
    graph_attrs=lambda attrs: dict(attrs),
    node_attrs=lambda attrs: json.loads(attrs["payload"]),
    edge_attrs=lambda attrs: json.loads(attrs["payload"]),
)
```

### Target files
- `src/codeintel/build/graphs/rx/serialization.py`
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/rx/store.py`

### Implementation checklist
- [x] Enforce dict[str, str] extractors for node-link JSON.
- [x] Embed graph metadata (policy/version) in `graph.attrs`.
- [x] Ensure round-trip preserves node/edge payloads.

### Status (completed in Scope 5)
- Node-link JSON payloads are JSON-encoded deterministically; legacy decode remains supported.
- Graph metadata is attached on store creation and preserved on load/serialize paths.

---

## Scope 6: Determinism + finalize contract enforcement

### Goal
Encode ordering/dedupe policy in the DSL and enforce canonical ordering in finalize.

### Code patterns

```python
import pyarrow.compute as pc

idx = pc.sort_indices(table, sort_keys=contract_sort_keys)
canon = table.take(idx)
```

```python
# Dedupe policy: keys + tie-breakers
sort_keys = [("src", "ascending"), ("dst", "ascending"), ("confidence", "descending")]
```

### Target files
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/graphs/runtime/context.py`

### Implementation checklist
- [ ] Represent ordering behavior in DSL nodes.
- [ ] Apply canonical sorting in finalize for CANONICAL tier.
- [ ] Move any order-dependent dedupe into finalize gate.
- [ ] Add contract metadata for determinism tier and ordering keys.

---

## Scope 7: Graph views + filtered analytics via rustworkx subgraphs

### Goal
Use rustworkx subgraph utilities rather than reconstructing filtered graphs.

### Code patterns

```python
subgraph, node_map = store.graph.subgraph_with_nodemap(selected_indices)
sub_store = RxGraphStore.from_rx_graph(subgraph, weight_policy=store.weight_policy)
```

### Target files
- `src/codeintel/build/graphs/compute/metrics/centrality.py`
- `src/codeintel/build/graphs/compute/metrics/components.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`

### Implementation checklist
- [x] Replace per-filter rebuild loops with subgraphs.
- [x] Use node maps to preserve stable ID mapping.
- [x] Normalize subgraph outputs via `stable_key`.

### Status (completed in Scope 7)
- Graph metric filters now use `subgraph_with_nodemap(..., preserve_attrs=True)`.

---

## Scope 8: Observability + provenance alignment (Acero + finalize)

### Goal
Leverage scan provenance and ensure structured artifacts drive validation rather than ad hoc prints.

### Code patterns

```python
scan = acero.Declaration(
    "scan",
    acero.ScanNodeOptions(
        dataset,
        implicit_ordering=True,
        require_sequenced_output=True,
    ),
)
```

```python
# Use provenance fields as tie-breakers for canonical sorting
sort_keys = [("file_id", "ascending"), ("fragment_index", "ascending"), ("batch_index", "ascending")]
```

### Target files
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/tabular/arrow_ops.py`

### Implementation checklist
- [ ] Ensure scan profiles surface provenance fields when requested.
- [ ] Use provenance in deterministic ordering where applicable.
- [ ] Treat finalize artifacts as primary outputs (no print debugging).
