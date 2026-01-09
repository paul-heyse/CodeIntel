# Rustworkx + Acero Unified Best-in-Class Implementation Plan

## Purpose
Deliver a single, Arrow-first graph assembly surface that feeds rustworkx through
contract-driven finalize boundaries, with deterministic ordering and structured
observability. This plan aligns rustworkx with the unified core columnar and
analytics alignment targets while maximizing performance, modularity, and reuse.

## Non-Negotiables (for this plan)
- Graph inputs are finalized against contracts before graph construction.
- Determinism tier is explicit and drives ordering/provenance behavior.
- Acero/DSL is the primary assembly lane; Python loops are reserved for
  rustworkx ingestion only.
- Graph metadata is persisted and round-trippable (node-link JSON).

---

## Scope 1 — Arrow-first graph assembly via Acero DSL + finalize gates

Status: Completed (finalize gates + determinism ordering enforced in graph views)

### Pattern
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeSpec
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.build.tabular.plan_ops import Plan
from codeintel.build.tabular.expr_vocab import E

ctx = ExecutionContext(determinism="canonical", provenance=True)
plan = Plan.scan(dataset, columns=columns).filter(E.is_valid("src_id"))
plan = plan.project({"src_id": E.field("src_id"), "dst_id": E.field("dst_id")})
plan = plan.order_by(sort_keys=[("src_id", "ascending"), ("dst_id", "ascending")])
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan),
    finalize=FinalizeSpec(
        table_key="graph.call_graph_edges",
        mode="tolerant",
        emit_artifacts=True,
    ),
    ctx=ctx,
)
edge_table = result.good
```

### Target files
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/finalize_ops.py`

### Checklist
- [x] Use `Plan.scan`/`Plan.table` + Acero nodes for graph edge/node assembly.
- [x] Enforce finalize gates for edge/node tables before rustworkx ingestion.
- [x] Apply canonical ordering in finalize when determinism is CANONICAL.
- [ ] Prefer `to_reader` for streaming until finalize boundaries.

---

## Scope 2 — Unified rustworkx GraphBuilder + bulk edge ingestion

### Pattern
```python
from codeintel.build.graphs.rx.build_from_edges import BuildStoreOptions, EdgeBuildSpec
from codeintel.build.graphs.rx.build_from_edges import build_store_from_edge_tuples
from codeintel.build.graphs.rx.policies import DEFAULT_NUMERIC_POLICY, weight_policy_for_kind

spec = EdgeBuildSpec(
    directed=True,
    weight_policy=weight_policy_for_kind(GraphKind.CALL_GRAPH),
    numeric_policy=DEFAULT_NUMERIC_POLICY,
)
options = BuildStoreOptions(
    stable_nodes=True,
    aggregate_edges=True,
    node_ids=node_ids,
    node_attrs=node_attrs,
    node_hint=len(node_ids),
    edge_hint=edge_table.num_rows,
)
store = build_store_from_edge_tuples(edge_rows, spec=spec, options=options)
```

### Target files
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/builders.py`

### Checklist
- [ ] Centralize graph construction through `EdgeBuildSpec` + bulk edge insertion.
- [ ] Ensure stable node ordering and weight aggregation are the default.
- [ ] Add hooks for node attrs derived from finalized tables.
- [ ] Standardize on edge tuple ingestion (`(src, dst, weight)` or inferred weight).

---

## Scope 3 — Determinism + ordering policy for graph inputs

Status: In progress (ordering metadata propagation still pending)

### Pattern
```python
from codeintel.core.schemas.primitives import resolve_stable_sort_keys
from codeintel.core.columnar.finalize_ops import FinalizeSpec

sort_keys = resolve_stable_sort_keys(schema)
spec = FinalizeSpec(
    table_key=table_key,
    mode="tolerant",
    emit_artifacts=True,
    order_by=tuple((key, "ascending") for key in sort_keys or ()),
)
```

### Target files
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/builders.py`

### Checklist
- [x] Canonical determinism enforces explicit ordering (contract sort keys + tie-breakers).
- [x] Use provenance columns as deterministic tie-breakers when required.
- [ ] Propagate ordering metadata through plan nodes to finalize.
- [x] Ensure graph builders only consume finalized (ordered) tables.

---

## Scope 4 — Rustworkx algorithm envelope + typed APIs

### Pattern
```python
from codeintel.build.graphs.rx.algos import GraphAlgoConfig, ensure_store
from codeintel.build.graphs.rx.weights import WeightSemantics
import rustworkx as rx

store = ensure_store(graph)
config = GraphAlgoConfig(weight_semantics=WeightSemantics.COST, rayon_threads=4)
paths = rx.digraph_dijkstra_shortest_paths(
    store.graph,
    source_idx,
    weight_fn=lambda payload: edge_cost_from_payload(
        payload,
        nan_policy=store.numeric_policy.nan_policy,
        semantics=config.weight_semantics or store.weight_policy.semantics,
        epsilon=config.weight_epsilon,
    ),
)
```

### Target files
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/rx/weights.py`
- `src/codeintel/build/graphs/compute/metrics/paths.py`
- `src/codeintel/build/graphs/compute/metrics/statistics.py`

### Checklist
- [ ] Route weighted algorithms through `GraphAlgoConfig` + weight semantics helpers.
- [ ] Prefer typed rustworkx APIs (`graph_*` / `digraph_*`) at call sites.
- [ ] Normalize outputs with stable ordering before emitting results.
- [ ] Centralize weight_fn construction and reuse across metrics modules.

---

## Scope 5 — Rustworkx primitives for components/condensation/subgraphs

### Pattern
```python
import rustworkx as rx
from codeintel.build.graphs.rx.normalize import stable_key

subgraph, _ = store.graph.subgraph_with_nodemap(node_indices, preserve_attrs=True)
condensed = rx.condensation(store.graph)
node_map = condensed.attrs.get("node_map")
layered = list(rx.layers(store.graph, first_layer=roots, index_output=True))
```

### Target files
- `src/codeintel/build/graphs/compute/metrics/components.py`
- `src/codeintel/build/graphs/compute/metrics/cfg.py`
- `src/codeintel/build/graphs/compute/metrics/statistics.py`
- `src/codeintel/build/graphs/compute/imports.py`
- `src/codeintel/build/graphs/compute/metrics/community.py`

### Checklist
- [ ] Replace bespoke SCC/condensation logic with `rx.condensation` and friends.
- [ ] Use `subgraph_with_nodemap` for filtered graph views (stable ordering).
- [ ] Use `rx.layers` / `rx.topological_generations` for DAG layer logic.
- [ ] Use `rx.union`/`rx.compose` for graph merges where applicable.

---

## Scope 6 — Serialization + metadata (node-link JSON)

### Pattern
```python
import rustworkx as rx
from codeintel.build.graphs.rx.metadata import GraphMetadata, apply_graph_metadata

apply_graph_metadata(graph, GraphMetadata(
    weight_policy=store.weight_policy.name,
    determinism_tier="canonical",
))
json_payload = rx.node_link_json(
    graph,
    graph_attrs=lambda attrs: {str(k): str(v) for k, v in attrs.items()},
    node_attrs=lambda payload: {"payload": json.dumps(payload, sort_keys=True)},
    edge_attrs=lambda payload: {"payload": json.dumps(payload, sort_keys=True)},
)
```

### Target files
- `src/codeintel/build/graphs/rx/serialization.py`
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/rx/store.py`

### Checklist
- [ ] Ensure metadata (weight policy, determinism tier, engine) is embedded in graph attrs.
- [ ] Use structured node/edge payload encoding for lossless node-link JSON.
- [ ] Enforce round-trip load/store for graph persistence and cache reuse.

---

## Scope 7 — Observability alignment for graph inputs

### Pattern
```python
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_reader

result = finalize_reader(
    reader,
    spec=FinalizeSpec(
        table_key="graph.import_graph_edges",
        mode="tolerant",
        emit_artifacts=True,
    ),
)
errors = result.errors
alignment = result.alignment
stats = result.stats
```

### Target files
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/graphs/engine/datasets.py`
- `src/codeintel/build/analytics/graphs/orchestrator.py`

### Checklist
- [ ] Persist finalize artifacts for graph inputs in analytics/graph pipelines.
- [ ] Surface provenance fields for deterministic tie-breaks in canonical tier.
- [ ] Record run metadata (determinism tier, scan profile) with graph outputs.

---

## Scope 8 — Runtime profile + threading convergence for rustworkx

### Pattern
```python
from codeintel.build.graphs.runtime.context import GraphContext
from codeintel.build.graphs.rx.algos import GraphAlgoConfig

config = GraphAlgoConfig(
    parallel_threshold=ctx.parallel_threshold,
    rayon_threads=ctx.rayon_threads,
    weight_semantics=ctx.weight_semantics,
)
```

### Target files
- `src/codeintel/build/graphs/runtime/context.py`
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/runtime/runtime.py`

### Checklist
- [ ] Align rustworkx threading controls with ExecutionContext/runtime profiles.
- [ ] Use deterministic seeds and parallel thresholds tied to runtime profile.
- [ ] Explicitly encode determinism tier into graph metadata.

---

## Sequencing Recommendation
1) Scope 1 (Arrow-first graph assembly + finalize gates)
2) Scope 2 (Unified GraphBuilder + bulk ingestion)
3) Scope 3 (Determinism + ordering policy)
4) Scope 4 (Typed algorithm envelope)
5) Scope 5 (Primitive replacements)
6) Scope 6 (Serialization + metadata)
7) Scope 7 (Observability alignment)
8) Scope 8 (Runtime profile convergence)

## Validation Guidance (tests are optional per request)
- Validate via `tools.quality_report` once implementation stabilizes.
- Prefer targeted module tests for graph pipelines when re-enabled.
