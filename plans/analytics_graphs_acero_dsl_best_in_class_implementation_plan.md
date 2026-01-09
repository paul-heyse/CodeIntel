# Analytics + Graphs Acero/DSL Best-in-Class Implementation Plan

## Objective
Deliver a fully plan-first, Acero/DSL-centric analytics + graphs architecture that
routes all relational logic through Plans, isolates kernel/external boundaries,
and enforces determinism + contracts at finalize.

## Compatibility with build_acero_dsl_schema_inference_plan.md
This plan is **complementary** and **non-conflicting**:
- It focuses on **analytics + graphs adoption** and **external plan boundaries**.
- It relies on the core plan/schema infrastructure from
  `plans/build_acero_dsl_schema_inference_plan.md` (Scopes 1-4, 8).
- It does **not** redefine plan schema propagation or the inference service; it
  consumes those capabilities.

---

## Scope 1 - Unified Plan Lane for Analytics + Graphs
**Goal**  
Every analytics/graph dataset is expressed as a `QuerySpec` and compiled by the
schema-default plan builder, with a single pipeline runner.

**Code pattern**
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
- `src/codeintel/build/analytics/graphs/*.py`
- `src/codeintel/build/analytics/functions/*.py`
- `src/codeintel/build/analytics/subsystems/*.py`
- `src/codeintel/build/graphs/engine/views.py`

**Implementation checklist**
- [ ] Route all scans through `QuerySpec` + `plan_from_schema_defaults`.
- [ ] Replace ad-hoc projections and filters in analytics modules.
- [ ] Ensure pipeline runner is the only plan execution entry point.

---

## Scope 2 - Reader-First + Finalize-Only Materialization
**Goal**  
Streaming execution everywhere; materialization only inside finalize.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.finalize_ops import finalize_reader, resolve_finalize_spec

reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
result = finalize_reader(reader, spec=resolve_finalize_spec(table_key))
table = result.good
```

**Target files**
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/build/graphs/assembly/finalize.py`
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/hamilton/native/graphs/*`

**Implementation checklist**
- [ ] Remove direct `to_table()` calls outside finalize.
- [ ] Ensure all readers are finalized via `finalize_reader`.
- [ ] Keep ordering/determinism enforcement in finalize only.

---

## Scope 3 - Kernel Lane Standardization (Rowsets + AST Worklists)
**Goal**  
Row-count-changing ops move into shared kernel helpers; AST parsing is driven
only by Arrow worklists and emits Arrow outputs.

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
from codeintel.build.analytics.parsing.worklists import build_function_ast_worklist

worklist = build_function_ast_worklist(goids_frame, repo=repo, commit=commit, ctx=ctx)
for row in iter_tuples(worklist.to_reader(), columns=("goid_h128", "rel_path")):
    parse_and_accumulate(...)
```

**Target files**
- `src/codeintel/build/analytics/parsing/worklists.py`
- `src/codeintel/build/analytics/ast_features/extract.py`
- `src/codeintel/build/analytics/functions/metrics.py`
- `src/codeintel/build/analytics/functions/function_contracts.py`
- `src/codeintel/build/analytics/graphs/config_data_flow.py`
- `src/codeintel/build/analytics/graphs/config_references.py`
- `src/codeintel/build/analytics/subsystems/affinity.py`

**Implementation checklist**
- [ ] Replace row loops with grouped rollups / explode kernels.
- [ ] Drive AST parsing exclusively from worklists.
- [ ] Emit Arrow outputs and finalize via analytics helpers.

---

## Scope 4 - Graph Analytics as External Plans
**Goal**  
Treat rustworkx execution as an external plan lane that consumes Arrow tables
and emits Arrow outputs, preserving determinism and schema inference.

**Code pattern**
```python
from codeintel.build.graphs.external_plan import run_rustworkx_external_plan
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.expr_vocab import E

edges_plan = (
    Plan.table(edge_table)
    .project(src=E.field("caller_goid_h128"), dst=E.field("callee_goid_h128"))
    .aggregate(keys=["src", "dst"], aggregates=[("src", "count", None, "weight")])
)
reader = run_rustworkx_external_plan(
    builder=build_call_graph_from_edges,
    args=(edges_plan,),
    kwargs={"directed": True},
    use_threads=ctx.use_threads,
)
```

**Target files**
- `src/codeintel/build/graphs/external_plan.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/analytics/subsystems/affinity.py`
- `src/codeintel/build/graphs/compute/metrics/*`

**Implementation checklist**
- [ ] Register rustworkx external runner.
- [ ] Build edge/node tables via Plan + finalize before rustworkx.
- [ ] Return `RecordBatchReader` from external plan and finalize outputs.

---

## Scope 5 - Graph Builder Unification (EdgeBuildSpec)
**Goal**  
All graph loaders use `EdgeBuildSpec + build_store_from_edge_tuples` with stable
node ordering and explicit weight policies.

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
options = BuildStoreOptions(stable_nodes=True, node_hint=200_000, edge_hint=2_000_000)
store = build_store_from_edge_tuples(edge_rows, spec=spec, options=options)
```

**Target files**
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/analytics/subsystems/affinity.py`

**Implementation checklist**
- [ ] Remove bespoke edge insertion loops.
- [ ] Provide stable node lists and weight policies.
- [ ] Normalize node attrs with `EdgeBuildSpec` hooks.

---

## Scope 6 - Typed Algorithm Envelope + Iterators
**Goal**  
All rustworkx algorithms go through typed wrappers with explicit weight
semantics, normalized outputs, and shared iterators.

**Code pattern**
```python
from codeintel.build.graphs.rx.algos import GraphAlgoConfig, digraph_katz_centrality_by_id
from codeintel.build.graphs.rx.algos import edge_cost_weight_fn, resolve_weight_context

config = GraphAlgoConfig(weight_semantics="cost")
context = resolve_weight_context(store, algo_config=config)
weight_fn = edge_cost_weight_fn(context=context)
scores = digraph_katz_centrality_by_id(store, weight_fn=weight_fn, algo_config=config)
```

**Target files**
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/rx/iterators.py`
- `src/codeintel/build/graphs/rx/normalize.py`
- `src/codeintel/build/graphs/compute/metrics/*`
- `src/codeintel/build/analytics/graphs/*`

**Implementation checklist**
- [ ] Replace direct rustworkx calls with typed wrappers.
- [ ] Normalize outputs via shared iterators.
- [ ] Apply weight semantics consistently.

---

## Scope 7 - Determinism + Serialization Metadata
**Goal**  
Determinism tier and ordering metadata are contract-driven and embedded in all
graph outputs, including node-link JSON.

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
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/rx/serialization.py`
- `src/codeintel/build/graphs/runtime/runtime.py`
- `src/codeintel/build/analytics/graphs/*.py`

**Implementation checklist**
- [ ] Persist determinism tier + ordering keys in metadata.
- [ ] Require metadata in node-link serialization.
- [ ] Ensure all graph exports attach metadata.

---

## Scope 8 - Schema Inference + Contract Reduction (Analytics/Graphs)
**Goal**  
Plan schema compiler is authoritative; output registry retains only contract
constraints (keys, dedupe, determinism policy).

**Code pattern**
```python
from codeintel.core.columnar.plan_schema import compile_plan_schema
from codeintel.core.schemas.table_schema import table_schema_from_arrow_schema

schema = compile_plan_schema(plan, inputs=input_schemas)
contract = table_schema_from_arrow_schema(schema)
```

**Target files**
- `src/codeintel/build/schemas/inference_service.py`
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/columnar/plan_schema.py`

**Implementation checklist**
- [ ] Remove explicit schema declarations where plan inference applies.
- [ ] Retain only contract constraints in output registry.
- [ ] Ensure schema serde preserves determinism + dedupe policy.

---

## Scope 9 - Observability + Provenance (Plan Runs)
**Goal**  
Every analytics/graph pipeline emits run manifests, finalize artifacts, and scan
telemetry with ordering metadata.

**Code pattern**
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
- `src/codeintel/build/hamilton/native/analytics/*.py`

**Implementation checklist**
- [ ] Attach telemetry for dataset-backed plans.
- [ ] Persist finalize artifacts beside outputs.
- [ ] Emit manifests for all analytics/graph outputs.

---

## Scope 10 - External Plan Registry + DSL Extensions
**Goal**  
External engines (rustworkx, substrait) are first-class plan runners.

**Code pattern**
```python
from codeintel.core.columnar.plan_ops import register_external_plan_runner

def rustworkx_runner(*, request: ExternalPlanRequest) -> pa.RecordBatchReader:
    ...

register_external_plan_runner("rustworkx", rustworkx_runner)
```

**Target files**
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/plan_kernels.py`
- `src/codeintel/build/graphs/external_plan.py`
- `src/codeintel/core/columnar/arrowdsl.py`

**Implementation checklist**
- [ ] Register external plan runners.
- [ ] Standardize external plan request/response contracts.
- [ ] Keep kernel helpers typed and shared.

