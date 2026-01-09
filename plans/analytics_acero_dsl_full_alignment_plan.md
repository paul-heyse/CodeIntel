# Analytics Acero/DSL Full Alignment Plan

## Objective
Make every analytics module in `src/codeintel/build/analytics` route through the
pyarrow Acero/DSL framework, with explicit plan-lane vs kernel-lane boundaries,
schema-driven determinism, and structured artifacts.

## Scope 1 - Unified Plan Lane Entry (QuerySpec + Plan Builder Everywhere)
**Goal**
Every analytics scan is expressed via `QuerySpec` and compiled by the centralized
plan builder, with schema-driven projection, provenance, and ordering policies.

**Code patterns**
```python
from codeintel.build.analytics.utilities.snapshot import (
    SnapshotContext,
    build_snapshot_query_spec,
)
from codeintel.build.analytics.utilities.pipeline import (
    AnalyticsPipelineRunRequest,
    run_analytics_pipeline,
)

spec = build_snapshot_query_spec(
    base_cols=("repo", "commit", "function_goid_h128"),
    context=SnapshotContext(repo=repo, commit=commit, ctx=ctx, table_key=table_key),
)
result = run_analytics_pipeline(
    AnalyticsPipelineRunRequest(
        source=dataset,
        spec=spec,
        table_key=table_key,
        ctx=ctx,
    )
)
table = result.good
```

**Target files**
- `src/codeintel/build/analytics/utilities/snapshot.py`
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/analytics/utilities/datasets.py`
- `src/codeintel/build/analytics/py_cpg_quality_report.py`
- `src/codeintel/build/analytics/scip_diagnostics_rollups.py`
- `src/codeintel/build/analytics/graphs/*.py`
- `src/codeintel/build/analytics/functions/*.py`
- `src/codeintel/build/analytics/subsystems/*.py`

**Implementation checklist**
- [ ] Ensure all dataset scans use `QuerySpec` and `plan_from_schema_defaults`.
- [ ] Remove ad-hoc scan/projection logic in analytics modules.
- [ ] Preserve runtime profile knobs via `ExecutionContext`.

---

## Scope 2 - Kernel Lane for Row-Changing Ops (Explode/Dedupe/Rollup)
**Goal**
Move row-changing operations into shared kernel helpers, keeping analytics
modules plan-first and decode-only at boundaries.

**Code patterns**
```python
from codeintel.core.columnar.plan_kernels import GroupedRollupSpec, grouped_rollup_table

rowset = grouped_rollup_table(
    table,
    spec=GroupedRollupSpec(
        keys=("function_goid_h128",),
        aggregates=[("edge_kind", "list", None, "edge_kind")],
        pre_sort_keys=(("function_goid_h128", "ascending"),),
    ),
    ctx=ctx,
)
```

```python
from codeintel.core.columnar.explode_ops import ExplodeSpec
from codeintel.core.columnar.plan_kernels import explode_edges_for_join
from codeintel.core.schemas.service import get_schema_service

exploded = explode_edges_for_join(
    table,
    spec=ExplodeSpec(list_column="extras.reference_paths", output_column="reference_path"),
    table_key="analytics.config_references",
    schema_service=get_schema_service(),
).table
```

**Target files**
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/graphs/config_references.py`
- `src/codeintel/build/analytics/functions/function_effects.py`
- `src/codeintel/build/analytics/subsystems/affinity.py`

**Implementation checklist**
- [ ] Replace list/dedupe/rollup logic with kernel helpers.
- [ ] Keep list ordering semantics in `pre_sort_keys`.
- [ ] Restrict Python loops to final decoding boundaries only.

---

## Scope 3 - Graph Analytics as External Plans (Arrow-First + Finalize Boundary)
**Goal**
Treat rustworkx graph operations as an ExternalPlan backend to the DSL, with
Arrow-first edge/node assembly + finalize boundaries before graph ingestion.

**Code patterns**
```python
from codeintel.build.graphs.external_plan import run_rustworkx_external_plan
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.plan_ops import Plan

edges_plan = (
    Plan.table(edge_table)
    .project(src=E.field("caller_goid_h128"), dst=E.field("callee_goid_h128"))
    .aggregate(keys=["src", "dst"], aggregates=[("src", "count", None, "weight")])
)
edges = ExecutionPlan.from_plan(edges_plan, determinism="canonical").to_table(ctx=ctx)
edges = finalize_table(
    edges,
    spec=FinalizeSpec(name="graph_edges", determinism="canonical"),
)

reader = run_rustworkx_external_plan(
    builder=build_call_graph_from_edges,
    args=(edges,),
    kwargs={"directed": True},
    use_threads=ctx.use_threads,
)
```

**Target files**
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/graph_metrics_ext.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`
- `src/codeintel/build/analytics/graphs/subsystem_graph_metrics.py`
- `src/codeintel/build/analytics/subsystems/affinity.py`
- `src/codeintel/build/analytics/compute/graphs/*.py`
- `src/codeintel/build/graphs/external_plan.py`
- `src/codeintel/build/graphs/builders.py`

**Implementation checklist**
- [ ] Register a rustworkx external plan runner.
- [ ] Build edge/node tables via Plan + finalize before rustworkx ingestion.
- [ ] Move graph computation behind the external plan interface.
- [ ] Return `RecordBatchReader` and finalize like any other plan output.
- [ ] Migrate remaining direct rustworkx usage in `analytics/subsystems/affinity.py` and
      `analytics/compute/graphs/*` to `run_rustworkx_external_plan`.

---

## Scope 4 - GraphBuilder Unification (EdgeBuildSpec + Stable Nodes)
**Goal**
All analytics graph loaders route through a single EdgeBuildSpec-based builder
with stable node ordering and bulk edge ingestion.

**Code patterns**
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

```python
spec = EdgeBuildSpec(
    directed=False,
    weight_policy=weight_policy_for_kind(GraphKind.CONFIG_BIPARTITE),
    numeric_policy=DEFAULT_NUMERIC_POLICY,
    node_attrs_fn=lambda node_id, side: {"bipartite": side},
)
options = BuildStoreOptions(stable_nodes=True)
store = build_store_from_edge_tuples(edge_rows, spec=spec, options=options)
```

**Target files**
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/analytics/subsystems/affinity.py`

**Implementation checklist**
- [ ] Consolidate graph loaders into `build_store_from_edge_tuples`.
- [ ] Supply stable node lists and capacity hints where available.
- [ ] Remove bespoke per-edge add loops in analytics graph loaders.
- [ ] Replace `add_graph_weight` flows in analytics subsystems (e.g., affinity) with
      `EdgeBuildSpec` + node attrs where needed.

---

## Scope 5 - Schema-Driven Defaults and Join-Safe Policies
**Goal**
Make schema plan policies the single source of truth for default projections,
join-safe allowlists, and canonical ordering.

**Code patterns**
```python
from codeintel.core.columnar.plan_builder import SchemaPlanDefaultsRequest, plan_from_schema_defaults
from codeintel.core.schemas.service import get_schema_service

plan = plan_from_schema_defaults(
    schema_service=get_schema_service(),
    request=SchemaPlanDefaultsRequest(
        table_key="analytics.function_metrics",
        dataset=dataset,
        predicate=spec.predicate,
        columns=spec.projection.columns(),
        ctx=ctx,
    ),
)
```

**Target files**
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/build/analytics/utilities/snapshot.py`
- `src/codeintel/build/analytics/utilities/catalogs.py`

**Implementation checklist**
- [ ] Add PlanPolicy for all analytics tables without defaults.
- [ ] Route snapshot helpers through schema-driven defaults.
- [ ] Ensure schema serde preserves plan_policy fields.
- [ ] Update snapshot helpers to use `plan_from_schema_defaults` for in-memory tables
      (replace `build_snapshot_plan` / `resolve_default_projection` in
      `snapshot.py` and `catalogs.py`).

---

## Scope 6 - Determinism Policy + Ordering Enforcement
**Goal**
Make determinism tier and canonical ordering entirely contract-driven; remove
manual sorting that exists only for determinism.

**Code patterns**
```python
from codeintel.build.analytics.utilities.finalize import finalize_analytics_result

result = finalize_analytics_result(table_key, table)
return result.good
```

```python
plan = plan.order_by(
    sort_keys=[
        ("function_goid_h128", "ascending"),
        ("block_idx", "ascending"),
    ]
)
plan = plan.aggregate(
    keys=[E.field("function_goid_h128")],
    aggregates=[("block_idx", "list", None, "block_idx")],
)
```

**Target files**
- `src/codeintel/build/analytics/cfg_dfg/cfg_core.py`
- `src/codeintel/build/analytics/cfg_dfg/dfg_core.py`
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/module_graph_metrics_ext.py`
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/runtime/runtime.py`

**Implementation checklist**
- [ ] Remove determinism-only sorts outside list semantics.
- [ ] Keep `order_by` only when list ordering is required.
- [ ] Ensure finalize owns canonical ordering for persistent outputs.
- [ ] Persist ordering keys + determinism tier in graph metadata.

---

## Scope 7 - Observability + Provenance Everywhere
**Goal**
All analytics pipelines emit FinalizeResult artifacts and run manifests
with scan telemetry + ordering metadata.

**Code patterns**
```python
from codeintel.core.columnar.arrowdsl import PipelineRunOptions, run_pipeline
from codeintel.core.columnar.run_manifest import run_manifest_options_for_context
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

telemetry = scan_telemetry_for_queryspec(dataset, spec=spec)
options = PipelineRunOptions(
    ctx=ctx,
    scan_telemetry=telemetry,
    manifest_options=run_manifest_options_for_context(
        ctx=ctx,
        ordering=plan.ordering,
        scan_telemetry=telemetry,
    ),
)
result = run_pipeline(plan=ExecutionPlan.from_plan(plan), finalize=finalize, options=options)
```

**Target files**
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/analytics/utilities/datasets.py`
- `src/codeintel/build/analytics/py_cpg_quality_report.py`
- `src/codeintel/build/analytics/scip_diagnostics_rollups.py`
- `src/codeintel/build/hamilton/native/analytics/*.py`

**Implementation checklist**
- [ ] Attach scan telemetry for dataset-backed plans.
- [ ] Persist finalize artifacts alongside analytics outputs.
- [ ] Emit run manifests for all analytics pipelines.
- [ ] Ensure Hamilton analytics outputs persist finalize artifacts and run manifests
      (route writes through analytics dataset helpers or a shared pipeline wrapper).

---

## Scope 8 - Typed Algorithm Envelope + Weight Semantics
**Goal**
All rustworkx algorithm calls go through typed wrappers with explicit weight
semantics and normalized output ordering.

**Code patterns**
```python
from codeintel.build.graphs.rx.algos import GraphAlgoConfig
from codeintel.build.graphs.rx.algos import edge_cost_weight_fn, resolve_weight_context
from codeintel.build.graphs.rx.algos import digraph_katz_centrality_by_id

config = GraphAlgoConfig(weight_semantics="cost")
context = resolve_weight_context(store, algo_config=config)
weight_fn = edge_cost_weight_fn(context=context)
scores = digraph_katz_centrality_by_id(store, weight_fn=weight_fn, algo_config=config)
```

**Target files**
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/rx/weights.py`
- `src/codeintel/build/graphs/compute/metrics/*`
- `src/codeintel/build/analytics/graphs/*`

**Implementation checklist**
- [ ] Add wrappers for all weighted algorithms used by analytics.
- [ ] Replace direct rustworkx calls with typed wrappers.
- [ ] Normalize outputs with stable ordering and NaN policy.

---

## Scope 9 - Rustworkx Primitives Adoption
**Goal**
Replace bespoke graph logic with rustworkx primitives for components, subgraphs,
layers, and merges.

**Code patterns**
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
- `src/codeintel/build/graphs/compute/metrics/dfg.py`
- `src/codeintel/build/graphs/compute/metrics/statistics.py`
- `src/codeintel/build/graphs/compute/metrics/community.py`
- `src/codeintel/build/graphs/compute/imports.py`

**Implementation checklist**
- [ ] Replace SCC/condensation logic with `rx.condensation`.
- [ ] Use `subgraph_with_nodemap` for filtered views.
- [ ] Use `rx.layers` / `rx.topological_generations` for DAG layers.
- [ ] Use `rx.union` / `rx.compose` for graph merges.
- [ ] Replace bespoke topological layering and edge assembly in
      `components.py` / `community.py` with rustworkx primitives.

---

## Scope 10 - Return-Type Normalization + Iterators
**Goal**
Use shared iterators and normalize rustworkx return types centrally.

**Code patterns**
```python
from codeintel.build.graphs.rx.iterators import iter_edge_id_weights
from codeintel.build.graphs.rx.normalize import sorted_mapping

weights = {src: 0.0 for src, _, _ in iter_edge_id_weights(store)}
weights = sorted_mapping(weights)
```

```python
from codeintel.build.graphs.rx.iterators import iter_incident_edges

for src_id, dst_id, weight in iter_incident_edges(store, node_id):
    ...
```

**Target files**
- `src/codeintel/build/graphs/rx/iterators.py`
- `src/codeintel/build/graphs/rx/normalize.py`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/analytics/cfg_dfg/helpers.py`

**Implementation checklist**
- [ ] Add missing iterators (edge index map, payload tuples, incident edges).
- [ ] Replace manual edge loops with iterators.
- [ ] Normalize nested mappings via shared helpers.

---

## Scope 11 - Serialization + Metadata (Node-Link JSON)
**Goal**
Deterministic, lossless serialization with explicit metadata across graph outputs.

**Code patterns**
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
- `src/codeintel/build/analytics/graphs/config_graph_metrics.py`
- `src/codeintel/build/analytics/graphs/subsystem_graph_metrics.py`

**Implementation checklist**
- [ ] Use typed node-link JSON for both directed and undirected graphs.
- [ ] Persist determinism tier + ordering keys + weight policy in attrs.
- [ ] Ensure round-trip preserves node IDs and payloads.
- [ ] Attach ordering keys/metadata for config bipartite + subsystem graph outputs.

---

## Scope 12 - AST/Parsing Boundaries as Kernel Lane
**Goal**
Keep AST parsing and traversal inside explicit kernel boundaries, fed by
plan-first worklists and emitting Arrow tables for finalize.

**Code patterns**
```python
worklist = grouped_rollup_table(
    scoped,
    spec=GroupedRollupSpec(
        keys=("function_goid_h128",),
        aggregates=[("rel_path", "min", None, "rel_path")],
    ),
    ctx=ctx,
)
for row in iter_tuples(worklist.to_reader(), columns=("function_goid_h128", "rel_path")):
    parse_and_accumulate(...)
```

**Target files**
- `src/codeintel/build/analytics/functions/function_contracts.py`
- `src/codeintel/build/analytics/functions/metrics.py`
- `src/codeintel/build/analytics/ast_features/extract.py`
- `src/codeintel/build/analytics/parsing/*.py`

**Implementation checklist**
- [ ] Drive AST work exclusively from Arrow worklists.
- [ ] Emit Arrow tables from parsing kernels; avoid row-dict pipelines.
- [ ] Finalize outputs via analytics finalize helpers.
- [ ] Replace remaining row-dict loops in parsing/metrics modules with
      grouped rollups + Arrow worklists before AST boundaries.

---

## Scope 13 - DSL Framework Extensions (Core)
**Goal**
Extend the DSL so all analytics patterns are expressible without ad-hoc logic.

**Code patterns**
```python
from codeintel.core.columnar.plan_ops import register_external_plan_runner

def rustworkx_runner(*, request: ExternalPlanRequest) -> pa.RecordBatchReader:
    # build graph, compute metrics, return reader
    ...

register_external_plan_runner("rustworkx", rustworkx_runner)
```

**Target files**
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/plan_kernels.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/schemas/primitives.py`
- `src/codeintel/build/graphs/external_plan.py`

**Implementation checklist**
- [ ] Add/extend external plan runner for graph engines.
- [ ] Standardize list ordering + rowset specs in kernel lane.
- [ ] Keep all kernel lane helpers typed and shared.
