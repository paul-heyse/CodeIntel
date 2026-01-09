# Acero/DSL Centralized Maximal Followup Implementation Plan

## Objective
Deliver a maximally centralized, deduplicated Acero/DSL architecture that unifies
ingestion, analytics, and rustworkx graph computation under a single plan lane
(ExecPlan) + kernel lane (row-changing ops) + finalize boundary, with schema-driven
defaults, deterministic ordering, and first-class observability.

## Source Plans
This plan operationalizes and sequences the scopes from:
- `plans/ingestion_acero_dsl_maximal_framework_alignment_plan.md`
- `plans/analytics_acero_dsl_full_alignment_plan.md`
- `plans/rustworkx_best_in_class_capabilities_implementation_plan.md`

## Non-Negotiables
- QuerySpec is the only scan surface.
- Plan lane only for Acero (scan/filter/project/join/aggregate/order_by).
- Kernel lane only for row-changing ops (explode/dedupe/rollup/winner selection).
- Finalize is the only materialization boundary.
- Schema policy is the single source of truth for projection, join-safe columns,
  and canonical ordering.
- Run manifests and scan telemetry are emitted for every finalize boundary.

---

## Scope 01 — Unified Pipeline Runner + Plan Builder (Core DSL)
**Goal**
Make `run_pipeline(...)` the only execution boundary and enforce plan construction
through the core builder API for both dataset scans and in-memory tables.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.plan_builder import build_table_plan

plan = build_table_plan(table=table)
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan),
    finalize=finalize_spec_for_table("core.modules", mode="tolerant"),
    options=PipelineRunOptions(ctx=execution_ctx),
)
rows = result.good
```

**Target files**
- `src/codeintel/core/columnar/arrowdsl.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/ingestion/compute/plan_surface.py`
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/graphs/engine/datasets.py`

**Implementation checklist**
- [ ] Enforce `ExecutionPlan + run_pipeline` as the only execution path.
- [ ] Remove or deprecate raw `Plan.to_table()` usage outside finalize.
- [ ] Route all in-memory plans through `build_table_plan(...)`.
- [ ] Ensure dataset scans use `build_query_plan_for_context(...)`.

---

## Scope 02 — QuerySpec Control Plane (Ingestion + Analytics)
**Goal**
Centralize all scan predicate/projection logic into QuerySpec helpers and
compile through schema-driven defaults.

**Code pattern**
```python
from codeintel.core.columnar.plan_builder import SchemaPlanDefaultsRequest, plan_from_schema_defaults
from codeintel.core.schemas.service import get_schema_service

plan = plan_from_schema_defaults(
    schema_service=get_schema_service(),
    request=SchemaPlanDefaultsRequest(
        table_key="core.scip_occurrences",
        dataset=dataset,
        predicate=spec.predicate,
        columns=spec.scan_columns(provenance=False),
        ctx=execution_ctx,
    ),
)
```

**Target files**
- `src/codeintel/ingestion/compute/queryspecs.py`
- `src/codeintel/ingestion/compute/plan_surface.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/build/analytics/utilities/snapshot.py`

**Implementation checklist**
- [ ] Make QuerySpec helpers the only predicate/projection constructors.
- [ ] Eliminate ad hoc projection lists in ingestion and analytics modules.
- [ ] Include provenance columns when determinism is canonical.

---

## Scope 03 — Kernel Lane Standardization (Explode / Dedupe / Rollup)
**Goal**
Ensure every row-changing operation routes through kernel helpers with
schema-aware join safety and deterministic tie handling.

**Code pattern**
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec
from codeintel.core.columnar.plan_kernels import explode_edges_for_join

result = explode_edges_for_join(
    table=edges,
    spec=ExplodeSpec(src_col="src_id", dst_list_col="dst_ids"),
    table_key="core.syntax_edges",
    schema_service=schema_service,
)
exploded = result.good
```

```python
from codeintel.core.columnar.plan_kernels import StableDedupeSpec, stable_dedupe_with_ties

deduped = stable_dedupe_with_ties(
    table,
    spec=StableDedupeSpec(
        key_columns=("repo", "commit", "symbol"),
        order_by=(("score", "descending"),),
        tie_breakers=(("rel_path", "ascending"), ("start_line", "ascending")),
    ),
)
```

```python
from codeintel.core.columnar.plan_kernels import GroupedRollupSpec, grouped_rollup_table

rollup = grouped_rollup_table(
    table,
    spec=GroupedRollupSpec(
        keys=("repo", "commit"),
        aggregates=(("severity", "count", None, "diagnostic_count"),),
        pre_sort_keys=(("repo", "ascending"), ("commit", "ascending")),
    ),
    ctx=execution_ctx,
)
```

**Target files**
- `src/codeintel/core/columnar/plan_kernels.py`
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/analytics/graphs/config_*`
- `src/codeintel/build/analytics/functions/function_effects.py`

**Implementation checklist**
- [ ] Use `explode_edges_for_join(...)` everywhere list payloads are flattened.
- [ ] Use stable dedupe + explicit tie breakers for all winner selection paths.
- [ ] Replace ad hoc `group_by().aggregate(...)` with `grouped_rollup_table(...)`.
- [ ] Pass schema-driven join-safe allowlists into join paths.

---

## Scope 04 — Ordering + Determinism Enforcement
**Goal**
Make ordering transitions explicit in plan metadata and enforce canonical ordering
at finalize boundaries based on schema policy.

**Code pattern**
```python
from codeintel.core.columnar.ordering import OrderingSpec
from codeintel.core.columnar.arrowdsl import ExecutionPlan

plan = plan.order_by(sort_keys=[("repo", "ascending"), ("commit", "ascending")])
execution_plan = ExecutionPlan.from_plan(plan, ordering=OrderingSpec.explicit(
    keys=(("repo", "ascending"), ("commit", "ascending")),
    reason="canonical ordering",
))
```

**Target files**
- `src/codeintel/core/columnar/ordering.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`

**Implementation checklist**
- [ ] Propagate ordering through join/aggregate/order_by in Plan ops.
- [ ] Enforce stable_sort_keys precedence in schema ordering policy.
- [ ] Require explicit order_by when canonical determinism is requested.
- [ ] Remove determinism-only sorts outside kernel lane.

---

## Scope 05 — Schema-Driven Defaults + Join-Safe Policies
**Goal**
Make schema metadata the sole authority for projection defaults and join-safe
allowlists across ingestion and analytics.

**Code pattern**
```python
from codeintel.core.schemas.primitives import PlanPolicy, TableSchema

TableSchema(
    ...,
    plan_policy=PlanPolicy(
        default_projection=("repo", "commit", "rel_path"),
        join_safe_columns=("repo", "commit", "rel_path"),
    ),
)
```

**Target files**
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/schemas/view_registry.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/queryspec.py`

**Implementation checklist**
- [ ] Populate PlanPolicy for every ingestion + analytics table.
- [ ] Ensure views preserve plan_policy overrides.
- [ ] Remove call-site projection defaults once schema policies exist.

---

## Scope 06 — External Plan Unification (Rustworkx Backend)
**Goal**
Treat rustworkx as an ExternalPlan backend returning readers, with finalize
as the boundary before graph ingestion.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.plan_ops import ExternalPlanRequest, ExternalPlanSpec

request = ExternalPlanRequest(
    spec=ExternalPlanSpec(engine="rustworkx", payload=payload),
    dataset=dataset,
    filter_expr=None,
    columns=None,
    scan_options=None,
    use_threads=execution_ctx.resolve_use_threads(),
)
plan = ExecutionPlan.from_external_plan(request)
```

**Target files**
- `src/codeintel/build/graphs/external_plan.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/external_plans.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/graphs/builders.py`

**Implementation checklist**
- [ ] Register rustworkx runner returning `RecordBatchReader` only.
- [ ] Finalize edge/node tables before rustworkx ingestion.
- [ ] Route analytics graph metrics through external plan runners.

---

## Scope 07 — Graph Builder Unification (EdgeBuildSpec Everywhere)
**Goal**
Centralize graph construction in a single `build_store_from_edge_tuples` path
with stable node ordering and capacity hints.

**Code pattern**
```python
from codeintel.build.graphs.rx.build_from_edges import BuildStoreOptions, EdgeBuildSpec
from codeintel.build.graphs.rx.build_from_edges import build_store_from_edge_tuples

spec = EdgeBuildSpec(directed=True, weight_policy=weight_policy, numeric_policy=numeric_policy)
options = BuildStoreOptions(node_hint=200_000, edge_hint=2_000_000, stable_nodes=True)
store = build_store_from_edge_tuples(edge_rows, spec=spec, options=options)
```

**Target files**
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/builders.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/analytics/graphs/*`

**Implementation checklist**
- [ ] Remove bespoke per-edge add loops in loaders.
- [ ] Pass node_ids/node_attrs explicitly when available.
- [ ] Use bulk edge insertion everywhere.

---

## Scope 08 — Typed Algorithm Envelope + Rustworkx Primitives
**Goal**
Route all rustworkx algorithms through typed wrappers with explicit weight
semantics and stable output normalization.

**Code pattern**
```python
from codeintel.build.graphs.rx.algos import GraphAlgoConfig
from codeintel.build.graphs.rx.algos import digraph_katz_centrality_by_id

config = GraphAlgoConfig(weight_semantics="cost")
scores = digraph_katz_centrality_by_id(store, algo_config=config)
```

```python
import rustworkx as rx

condensed = rx.condensation(store.graph)
layers = rx.layers(condensed)
```

**Target files**
- `src/codeintel/build/graphs/rx/algos.py`
- `src/codeintel/build/graphs/rx/weights.py`
- `src/codeintel/build/graphs/compute/metrics/*`
- `src/codeintel/build/analytics/graphs/*`

**Implementation checklist**
- [ ] Add wrappers for missing weighted algorithms (HITS/Katz/transitivity/etc.).
- [ ] Replace bespoke SCC/condensation/subgraph logic with rustworkx primitives.
- [ ] Normalize outputs with shared iterators and sort helpers.

---

## Scope 09 — Serialization + Metadata (Node-Link JSON)
**Goal**
Produce deterministic graph serialization with explicit metadata for determinism
and ordering keys.

**Code pattern**
```python
from codeintel.build.graphs.rx.metadata import GraphMetadata, apply_graph_metadata
from codeintel.build.graphs.rx.serialization import dumps_node_link_json

apply_graph_metadata(store.graph, GraphMetadata(weight_policy="strength", determinism_tier="canonical"))
payload = dumps_node_link_json(store.graph, require_metadata=True)
```

**Target files**
- `src/codeintel/build/graphs/rx/metadata.py`
- `src/codeintel/build/graphs/rx/serialization.py`
- `src/codeintel/build/graphs/runtime/runtime.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`

**Implementation checklist**
- [ ] Persist determinism + ordering keys on all stored graphs.
- [ ] Require metadata for serialization outputs.
- [ ] Validate round-trip stability for node IDs and attrs.

---

## Scope 10 — Observability + Telemetry Unification
**Goal**
Ensure every pipeline emits scan telemetry and run manifests with ordering and
profile metadata.

**Code pattern**
```python
from codeintel.core.columnar.run_manifest import run_manifest_options_for_context
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

telemetry = scan_telemetry_for_queryspec(dataset, spec=spec)
options = PipelineRunOptions(
    ctx=execution_ctx,
    scan_telemetry=telemetry,
    manifest_dir=manifest_dir,
    manifest_options=run_manifest_options_for_context(
        ctx=execution_ctx,
        ordering=plan.ordering,
        scan_telemetry=telemetry,
    ),
)
```

**Target files**
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/analytics/utilities/pipeline.py`
- `src/codeintel/build/hamilton/native/ingestion/manifesting.py`
- `src/codeintel/build/graphs/validation/runner.py`

**Implementation checklist**
- [ ] Propagate telemetry into PipelineRunOptions everywhere.
- [ ] Emit manifests for ingestion + analytics pipelines.
- [ ] Persist ordering metadata for deterministic outputs.

---

## Scope 11 — Performance + Capacity Hints
**Goal**
Improve throughput by preallocating graph storage and reducing Python loops.

**Code pattern**
```python
from codeintel.build.graphs.rx.build_from_edges import BuildStoreOptions

options = BuildStoreOptions(node_hint=500_000, edge_hint=5_000_000, stable_nodes=True)
```

**Target files**
- `src/codeintel/build/graphs/rx/build_from_edges.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/analytics/graphs/graph_metrics.py`
- `src/codeintel/build/analytics/graphs/symbol_graph_metrics.py`

**Implementation checklist**
- [ ] Provide capacity hints at all graph construction sites.
- [ ] Use aggregated edge rows to minimize insertion volume.
- [ ] Remove residual per-edge loops in analytics graph loaders.

---

## Validation Gates (When Tests Resume)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted pytest subsets for modified analytics, ingestion, and graph modules.

