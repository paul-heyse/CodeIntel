# Build Acero DSL Schema Inference Plan

## Goals

- Make Arrow Acero and the DSL the primary compute surface for build and ingestion.
- Infer output schemas from plan graphs instead of declaring them manually.
- Keep finalize as the only materialization boundary and enforce contracts there.

## Status Summary

- Scope 1: complete (plan schema propagation + ordering metadata in Plan).
- Scope 2: in progress (plan schema compiler wired; output registry reduction pending).
- Scope 3: in progress (graph nodes + core ingestion nodes now return Plan; remaining Hamilton nodes still return tables).
- Scope 4: in progress (reader-first execution supported; remaining to_table helpers still exist).
- Scope 5: in progress (explode kernels centralized; remaining row-loop migrations pending).
- Scope 6: in progress (HashJoinSpec in DSL; legacy joins and aggregates pending).
- Scope 7: in progress (ingestion plan surface exists; full adoption pending).
- Scope 8: in progress (ordering/provenance infrastructure in place; call sites still need updates).
- Scope 9: in progress (analytics conversions and rustworkx boundaries pending).
- Scope 10: in progress (Substrait runner exists; schema inference alignment pending).

## Scope 1: Plan schema propagation in the DSL

Intent: Extend Plan operations to carry output schema and ordering metadata so plans can be statically analyzed.

Code pattern
```python
# src/codeintel/core/columnar/plan_ops.py
from codeintel.core.columnar.plan_schema import infer_project_schema

def project(self, expressions, *, names=None) -> Plan:
    expr_list, out_names = _normalize_project_args(expressions, names)
    options = acero.ProjectNodeOptions(expr_list, names=out_names)
    decl = acero.Declaration("project", options, inputs=[self.declaration])
    schema = infer_project_schema(self.schema, expr_list, out_names)
    ordering = _project_ordering(self._resolved_ordering(), expressions=expr_list, names=out_names)
    return Plan(decl, schema=schema, ordering=ordering)
```

Targets
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/plan_schema.py` (new)

Checklist
- [x] Add schema propagation for project, filter, aggregate, hash_join, order_by.
- [x] Persist OrderingSpec and determinism hints in Plan.
- [x] Ensure schema inference works for expression mappings and explicit names.

## Scope 2: Plan schema compiler and contract integration

Intent: Compile plan graphs into TableSchema and use that as the primary schema source.

Code pattern
```python
# src/codeintel/core/columnar/plan_schema.py
def compile_plan_schema(plan: Plan, *, inputs: Mapping[str, pa.Schema]) -> TableSchema:
    schema = resolve_plan_schema(plan, inputs=inputs)
    return table_schema_from_arrow_schema(schema)
```

Targets
- `src/codeintel/core/columnar/plan_schema.py` (new)
- `src/codeintel/build/schemas/inference_service.py`
- `src/codeintel/core/schemas/output_registry.py`

Checklist
- [x] Add a plan schema compiler that accepts input schemas and Plan graphs.
- [x] Wire compiler into inference service for plan backed outputs.
- [ ] Reduce output registry to contract constraints only (keys, ordering, determinism).

## Scope 3: Plan first Hamilton DAG

Intent: Make Hamilton nodes return Plan or ExecutionPlan instead of materialized tables.

Code pattern
```python
# src/codeintel/build/hamilton/native/graphs/call_graph.py
def call_graph_edges_plan(q__graph__call_edges: InferableTabularInput) -> Plan:
    table = tabular_to_table(q__graph__call_edges)
    return Plan.table(table).project({"src": E.field("src"), "dst": E.field("dst")})
```

Targets
- `src/codeintel/build/hamilton/native/graphs/*`
- `src/codeintel/build/hamilton/native/ingestion/*`
- `src/codeintel/build/hamilton/native/analytics/*`

Checklist
- [x] Replace materialized table outputs with Plan outputs in graph nodes (call_graph/import_graph/symbol_use/cfg_dfg).
- [x] Replace materialized table outputs with Plan outputs in core ingestion nodes (scip_resolution/syntax_augment/syntax_enrich).
- [ ] Replace materialized table outputs with Plan outputs in remaining Hamilton nodes.
- [x] Allow Plan/ExecutionPlan to flow through inference + materialization.
- [ ] Keep finalize and materialization in the target nodes only.
- [ ] Ensure plan ordering metadata is preserved end to end.

## Scope 4: Reader first execution and finalize boundaries

Intent: Use to_reader by default and materialize only inside finalize.

Code pattern
```python
# src/codeintel/core/columnar/arrowdsl.py
reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
result = finalize_reader(reader, spec=resolve_finalize_spec(table_key))
```

Targets
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/build/graphs/assembly/finalize.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/hamilton/native/graphs/*`

Checklist
- [ ] Replace plan to_table helpers with reader to finalize flow.
- [ ] Enforce finalize as the only place that calls reader.read_all().
- [x] Align determinism tier and ordering keys inside finalize.

## Scope 5: Kernel lane for list explode edge builders

Intent: Standardize list explode and list alignment validation as kernel utilities.

Code pattern
```python
# src/codeintel/build/tabular/explode_ops.py
def explode_edges(table: pa.Table, *, src_col: str, dst_list_col: str) -> pa.Table:
    parent_idx = pc.list_parent_indices(table[dst_list_col])
    return pa.table(
        {
            "src": pc.take(table[src_col], parent_idx),
            "dst": pc.list_flatten(table[dst_list_col]),
        }
    )
```

Targets
- `src/codeintel/build/tabular/explode_ops.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`

Checklist
- [x] Centralize list explode and list alignment validation.
- [ ] Replace row loops building edges with list explode kernels.
- [ ] Add tolerant error routing for misaligned list payloads.

## Scope 6: Join and aggregate normalization in Acero

Intent: Replace ad hoc join and aggregate patterns with HashJoinSpec and AggregateNodeOptions.

Code pattern
```python
# src/codeintel/core/columnar/plan_ops.py
plan = Plan.table(left).hash_join(
    right=Plan.table(right),
    spec=HashJoinSpec(
        how="inner",
        left_keys=["symbol_id"],
        right_keys=["symbol_id"],
        left_output=["symbol_id", "src"],
        right_output=["dst"],
    ),
)
```

Targets
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/build/hamilton/native/graphs/cpg2/planes/*`
- `src/codeintel/build/analytics/graphs/*`

Checklist
- [x] Consolidate joins around HashJoinSpec in plan ops.
- [ ] Normalize aggregates to explicit output names.
- [ ] Remove Python joins and row grouping loops.

## Scope 7: Ingestion plan unification

Intent: Ensure ingestion outputs are constructed via QuerySpec and Plan pipelines.

Code pattern
```python
# src/codeintel/ingestion/compute/plan_surface.py
spec = build_ingest_query_spec(table_key, request)
plan = build_query_plan_for_context(dataset, spec=spec, ctx=resolved_ctx)
```

Targets
- `src/codeintel/ingestion/compute/*`
- `src/codeintel/ingestion/ports/*`

Checklist
- [ ] Drive ingestion reads through QuerySpec and Plan.
- [ ] Remove direct table materialization in ingestion compute.
- [ ] Align ingestion plans with execution context profiles.

## Scope 8: Ordering and provenance enforcement

Intent: Make deterministic ordering explicit and use provenance fields as tie breakers.

Code pattern
```python
# src/codeintel/core/columnar/plan_ops.py
plan = Plan.scan(dataset, implicit_ordering=True, require_sequenced_output=True)
plan = plan.order_by(sort_keys=[("repo", "ascending"), ("commit", "ascending")])
```

Targets
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/streaming.py`

Checklist
- [x] Propagate OrderingSpec through plan nodes.
- [ ] Use scan ordering settings only when canonical determinism is required.
- [x] Tie break canonical ordering with provenance fields.

## Scope 9: Analytics and graph pipeline conversion

Intent: Convert analytics and graph pipelines to Plan first and keep graph metrics in rustworkx only where required.

Code pattern
```python
# src/codeintel/build/analytics/graphs/config_graph_metrics.py
reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
table = finalize_reader(reader, spec=resolve_finalize_spec(table_key)).good
```

Targets
- `src/codeintel/build/analytics/graphs/*`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/analytics/cfg_dfg/*`

Checklist
- [ ] Replace iter_rows loops with plan aggregates where possible.
- [ ] Keep rustworkx algorithms as post plan computation only.
- [ ] Align graph dataset outputs with plan schema inference.

## Scope 10: Optional Substrait plan integration

Intent: Enable Substrait as an optional authoring surface for plans when needed.

Code pattern
```python
# src/codeintel/core/columnar/plan_ops.py
request = ExternalPlanRequest(plan_bytes=substrait_bytes, use_threads=True)
plan = ExecutionPlan.from_external_plan(request)
```

Targets
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/external_plans.py`

Checklist
- [x] Add a Substrait plan runner for external plans.
- [x] Wire into ExecutionPlan for optional use.
- [ ] Keep plan schema inference aligned with Substrait outputs.

## Rollout order

1) Plan schema propagation and compiler (Scope 1 and 2). Done, except output registry reduction.
2) Plan first Hamilton and reader first finalize (Scope 3 and 4). In progress (graphs + core ingestion nodes done).
3) Kernel explode utilities and join normalization (Scope 5 and 6). In progress.
4) Ingestion unification and ordering enforcement (Scope 7 and 8). In progress.
5) Analytics conversion and optional Substrait integration (Scope 9 and 10). In progress.
