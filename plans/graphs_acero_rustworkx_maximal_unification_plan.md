# Graphs: Acero DSL + Rustworkx Maximal Unification Plan

## Purpose
Reframe the entire graph stack around a single Arrow Acero DSL plan lane, a small
kernel lane for list explode + hashing, and a unified rustworkx ingestion and
algorithm envelope. The goal is to minimize bespoke Python loops and converge on
contract-driven finalize boundaries, deterministic ordering, and reusable graph
operators.

## Non-negotiables
- All graph inputs are produced by Acero plans or kernel helpers, not Python loops.
- Finalize gates enforce contract alignment and ordering before rustworkx ingest.
- Determinism tier is explicit and propagated across scan, plan, finalize, and graph metadata.
- Rustworkx usage is centralized and typed (PyGraph/PyDiGraph and graph_* / digraph_* APIs).
- Ordering is derived from contract keys with provenance tie-breakers for canonical tier.

---

## Scope 1 - Central Graph Plan Library (Acero DSL first)

**Goal**
Create a single module that exposes canonical plan patterns for graph assembly
(scan, project, filter, hash_join, aggregate, order_by). Every graph producer
must compose from this surface rather than hand-writing plan sequences.

**Pattern**
```python
from codeintel.build.graphs.assembly.plan_surface import graph_plan
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table

plan = (
    graph_plan.scan(dataset, columns=columns)
    .filter(graph_plan.expr.is_valid("src_id"))
    .project({"src_id": graph_plan.expr.field("src_id"),
              "dst_id": graph_plan.expr.field("dst_id")})
    .hash_join(spec=graph_plan.hash_join_spec("inner", keys=("src_id", "dst_id")))
)
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan, determinism=determinism),
    finalize=finalize_spec_for_table(table_key, determinism=determinism, mode="tolerant"),
    ctx=execution_ctx,
)
```

**Target files**
- New: `src/codeintel/build/graphs/assembly/plan_surface.py`
- New: `src/codeintel/build/graphs/assembly/plan_specs.py`
- Touch: `src/codeintel/core/columnar/arrowdsl.py`
- Touch: `src/codeintel/build/graphs/engine/views.py`

**Checklist**
- [ ] Implement graph_plan wrappers for scan/table, filter, project, aggregate, hash_join.
- [ ] Encode OrderingSpec and determinism tier on all plan nodes.
- [ ] Expose prebuilt join specs and common projections for graph edges/nodes.
- [ ] Document plan patterns for call/import/symbol/cfg/cdg/dfg producers.

---

## Scope 2 - Graph Kernel Library (explode, aligned lists, hashing)

**Goal**
Provide a small set of Arrow compute kernels that replace all Python row loops
for edge and attribute expansion.

**Pattern**
```python
from codeintel.build.graphs.assembly.kernels import explode_edges

edge_table = explode_edges(
    table=table,
    src_col="src_id",
    dst_list_col="dst_ids",
    repeat_cols=("repo", "commit", "file_id"),
)
```

**Target files**
- New: `src/codeintel/build/graphs/assembly/kernels.py`
- Touch: `src/codeintel/core/columnar/expr_vocab.py`
- Touch: `src/codeintel/core/columnar/kernels.py`

**Checklist**
- [ ] Implement list explode (list_parent_indices + list_flatten + take).
- [ ] Add aligned list validation for per-edge attributes (length checks).
- [ ] Provide deterministic hash ID helpers (hash_struct_ordinal).
- [ ] Provide stable sort helper that accepts contract keys + provenance tie-breakers.

---

## Scope 3 - Finalize Gate Wrapper for Graph Inputs

**Goal**
Centralize the "plan -> execute -> finalize -> artifacts" boundary for all graph
tables, with a single call site and consistent metadata.

**Pattern**
```python
from codeintel.build.graphs.assembly.finalize import finalize_graph_plan

result = finalize_graph_plan(
    plan=plan,
    table_key="graph.call_graph_edges",
    determinism=determinism,
    ctx=execution_ctx,
    emit_artifacts=True,
)
edge_table = result.good
```

**Target files**
- New: `src/codeintel/build/graphs/assembly/finalize.py`
- Touch: `src/codeintel/build/graphs/engine/datasets.py`
- Touch: `src/codeintel/build/graphs/engine/views.py`

**Checklist**
- [ ] Implement finalize_graph_plan wrapper (Acero plan + finalize spec + artifacts).
- [ ] Enforce canonical ordering based on contract keys in finalize.
- [ ] Persist artifacts and run metadata (determinism, runtime profile, scan profile).
- [ ] Use reader-first flow until finalize to preserve streaming.

---

## Scope 4 - Unified Rustworkx GraphBuilder (single ingestion path)

**Goal**
Remove bespoke graph construction loops in `builders.py` and enforce EdgeBuildSpec
ingestion for all graph types.

**Pattern**
```python
from codeintel.build.graphs.rx.build_from_edges import build_store_from_edge_tuples
from codeintel.build.graphs.rx.policies import weight_policy_for_kind

store = build_store_from_edge_tuples(
    edge_rows,
    spec=edge_spec,
    options=build_options,
)
```

**Target files**
- Touch: `src/codeintel/build/graphs/builders.py`
- Touch: `src/codeintel/build/graphs/rx/build_from_edges.py`
- Touch: `src/codeintel/build/graphs/engine/views.py`

**Checklist**
- [ ] Provide a single ingestion entry point for edge tables and node attrs.
- [ ] Ensure stable node ordering and aggregate edges by default.
- [ ] Accept RecordBatchReader inputs to avoid materialization where possible.
- [ ] Remove any remaining per-edge Python loops in graph builders.

---

## Scope 5 - Migrate Graph Producers to Plan + Kernel Lane

**Goal**
Convert graph producers to Acero plans and kernel explode helpers, eliminating
Python loops, direct `tabular_to_table`, and `iter_rows` usage.

**Pattern**
```python
plan = graph_plan.scan(dataset, columns=columns)
plan = graph_plan.project_edge_lists(plan, src="caller_goid_h128", dst_list="callee_ids")
edges = explode_edges(plan_to_table(plan), src_col="caller_goid_h128", dst_list_col="callee_ids")
```

**Target files**
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`
- `src/codeintel/build/hamilton/native/graphs/import_graph.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/cdg.py`
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
- `src/codeintel/build/hamilton/native/graphs/cpg/edges.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/*.py`

**Checklist**
- [ ] Replace iter_rows loops with Plan + kernel explode outputs.
- [ ] Normalize key casting and non-null gating before join and explode.
- [ ] Use HashJoinSpec with explicit key types, projections, and ordering.
- [ ] Ensure each producer finishes with finalize_graph_plan before ingestion.

---

## Scope 6 - Analytics Graph Pipelines in Columnar Form

**Goal**
Convert analytics graph pipelines to columnar row assembly without `iter_rows`
loops, using columnar buffers and Acero plans for table shaping.

**Pattern**
```python
from codeintel.core.columnar.rows import columnar_batch_collector_for_table_key

collector = columnar_batch_collector_for_table_key("analytics.graph_metrics_modules")
collector.extend(rows_dict)
table = collector.to_table()
```

**Target files**
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/graphs/subsystem_agreement.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/graph_metrics_ext.py`

**Checklist**
- [ ] Replace iter_rows usage with columnar collectors or Plan-based shaping.
- [ ] Keep graph inputs finalized and ordered before metrics ingestion.
- [ ] Ensure outputs are materialized using columnar buffers, not row dicts.

---

## Scope 7 - Determinism, Ordering, Runtime Profile Propagation

**Goal**
Guarantee determinism tier behavior is encoded in plan metadata, finalize gates,
and graph metadata. Ordering behavior should be explicit and derived from contracts.

**Pattern**
```python
spec = finalize_spec_for_table(
    table_key,
    determinism="canonical",
    order_by=(("repo", "ascending"), ("commit", "ascending"), ("src", "ascending")),
)
```

**Target files**
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/build/graphs/runtime/context.py`
- `src/codeintel/build/graphs/engine/views.py`

**Checklist**
- [ ] Embed OrderingSpec in plan nodes for canonical tier.
- [ ] Enforce contract key prefix for canonical ordering in finalize.
- [ ] Use provenance tie-breakers only when canonical tier requires it.
- [ ] Persist determinism tier and scan profile in graph metadata.

---

## Scope 8 - Rustworkx Algorithm Envelope and Weight Semantics

**Goal**
Standardize all algorithm usage on typed rustworkx APIs with explicit weight
semantics and stable output ordering.

**Pattern**
```python
paths = rx.digraph_dijkstra_shortest_paths(
    store.graph,
    source_idx,
    weight_fn=weight_fn,
)
```

**Target files**
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/rx/weights.py`
- `src/codeintel/build/graphs/compute/metrics/*.py`

**Checklist**
- [ ] Route all weighted algorithms through GraphAlgoConfig.
- [ ] Use digraph_* / graph_* typed APIs only.
- [ ] Normalize algorithm outputs using stable ordering utilities.

---

## Scope 9 - Serialization and Graph Metadata

**Goal**
Ensure node-link JSON and graph metadata are consistent, round-trip safe, and
include determinism and runtime metadata.

**Pattern**
```python
payload = rx.node_link_json(
    store.graph,
    node_attrs=lambda node: {"payload": json.dumps(node, sort_keys=True)},
    edge_attrs=lambda edge: {"payload": json.dumps(edge, sort_keys=True)},
)
```

**Target files**
- `src/codeintel/build/graphs/rx/serialization.py`
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/runtime/runtime.py`

**Checklist**
- [ ] Ensure metadata embeds determinism tier and scan profile.
- [ ] Preserve node/edge payloads losslessly through serialization.
- [ ] Maintain cache compatibility with graph metadata versioning.

---

## Sequencing Recommendation
1) Scope 1-3: Plan surface + kernel library + finalize wrapper
2) Scope 4-5: Unified GraphBuilder + producer migration
3) Scope 6: Analytics pipelines in columnar form
4) Scope 7-9: Determinism propagation + algorithm envelope + serialization

## Validation Guidance (tests optional)
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  after each phase to keep ruff/pyright/pyrefly aligned.
- Prefer targeted graph module validation when tests are re-enabled.
