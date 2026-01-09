# Ingestion Acero DSL Plan-First Implementation Plan

This plan defines the ingestion-specific migration to a fully Acero + DSL driven architecture.
It is complementary to `plans/build_acero_dsl_schema_inference_plan.md` and avoids duplicating
the core DSL/schema inference work already tracked there.

## Relationship to build_acero_dsl_schema_inference_plan.md

This plan assumes the following build plan items are in progress or completed:
- Plan schema propagation + compiler (Scopes 1-2 in the build plan).
- Plan-first Hamilton nodes and reader-first finalize boundaries (Scopes 3-4).
- Ordering/determinism metadata in Plan + finalize (Scope 8).
- QuerySpec and scan control plane foundations (Scope 7 + Chapter 6-7 guidance).

This plan focuses on ingestion-only work: ingestion compute modules, ingestion DAG nodes,
ingestion tooling outputs, and ingestion-specific contracts/manifesting.

## Scope 01: Plan-first ingestion sources (reader-friendly plan sources)

Intent: Ensure ingestion sources can enter the plan lane without materializing tables, and
standardize the reader -> Plan surface for ingestion-produced outputs.

Code pattern
```python
from codeintel.core.columnar.ordering import OrderingSpec
from codeintel.core.columnar.plan_ops import Plan

reader = collector.to_reader()
plan = Plan.reader_source(
    reader,
    ordering=OrderingSpec.implicit(reason="ingest reader source"),
)
```

Targets
- `src/codeintel/core/columnar/plan_ops.py` (add Plan.reader_source or equivalent)
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/ingestion/compute/plan_surface.py`
- `src/codeintel/ingestion/compute/*_extract.py`
- `src/codeintel/build/hamilton/native/ingestion/*`

Checklist
- [ ] Add a reader-backed plan source in the DSL (no implicit table materialization).
- [ ] Migrate ingestion extract steps to return Plan/ExecutionPlan instead of tables.
- [ ] Ensure reader-backed plans carry schema + ordering metadata.

## Scope 02: QuerySpec standardization for ingestion scans

Intent: All dataset-backed ingestion reads go through QuerySpec so pushdown, projection,
and provenance are consistent and observable.

Code pattern
```python
from codeintel.core.columnar.plan_ops import build_query_plan_for_context
from codeintel.ingestion.compute.queryspecs import build_ingest_query_spec

spec = build_ingest_query_spec("core.repo_map", request)
plan = build_query_plan_for_context(dataset, spec=spec, ctx=ctx)
```

Targets
- `src/codeintel/ingestion/compute/plan_surface.py`
- `src/codeintel/ingestion/compute/queryspecs.py`
- `src/codeintel/ingestion/compute/repo_scan.py`
- `src/codeintel/ingestion/compute/config_ingest.py`
- `src/codeintel/ingestion/compute/tests_ingest.py`
- `src/codeintel/ingestion/compute/typing_ingest.py`

Checklist
- [ ] Replace remaining dataset scans with QuerySpec + build_query_plan_for_context.
- [ ] Use a single scan-columns builder (projection + provenance in one place).
- [ ] Capture scan telemetry for ingestion scans and propagate to manifests.

## Scope 03: Reader-first execution and finalize-only materialization

Intent: Ingestion pipelines emit readers, and only finalize triggers materialization.

Code pattern
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.finalize_ops import finalize_reader, resolve_finalize_spec

reader = ExecutionPlan.from_plan(plan).to_reader(ctx=ctx)
result = finalize_reader(reader, spec=resolve_finalize_spec("core.modules"))
```

Targets
- `src/codeintel/ingestion/compute/base.py`
- `src/codeintel/build/hamilton/native/ingestion/manifesting.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`

Checklist
- [ ] Remove any residual plan->table materialization before finalize.
- [ ] Ensure finalize is the only place `read_all()` or table materialization occurs.
- [ ] Standardize reader-first finalization in ingestion DAG targets.

## Scope 04: Kernel lane for list explode + alignment policies

Intent: Centralize list explode and list alignment validation as kernel helpers with
explicit null-list policies and consistent error artifact routing.

Code pattern
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges_with_aligned_lists

spec = ExplodeSpec(
    list_col="params",
    null_list_policy="empty",
    null_child_policy="error",
)
edges = explode_edges_with_aligned_lists(table, spec=spec)
```

Targets
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/docstrings_extract.py`
- `src/codeintel/ingestion/compute/inspect_extract.py`
- `src/codeintel/ingestion/compute/tree_sitter_index.py`
- `src/codeintel/ingestion/scip/rows.py`

Checklist
- [ ] Replace list-bearing row loops with explode kernels.
- [ ] Define null-list policies per ingestion table (hard-error vs tolerate).
- [ ] Route list alignment errors to finalize artifacts and ingest manifests.

## Scope 05: Join normalization and join-safe projections

Intent: Replace ad hoc joins with HashJoinSpec and enforce join-safe projections before joins.

Code pattern
```python
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan
from codeintel.core.columnar.join_safe import join_safe_projection

left = left.project(join_safe_projection(left.schema, keys=("repo", "rel_path")))
plan = left.hash_join(
    right=right,
    spec=HashJoinSpec(
        how="inner",
        left_keys=["repo", "rel_path"],
        right_keys=["repo", "rel_path"],
        left_output=["repo", "rel_path", "node_id"],
        right_output=["metrics"],
    ),
)
```

Targets
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/ingestion/compute/typing_ingest.py`
- `src/codeintel/ingestion/compute/tests_ingest.py`

Checklist
- [ ] Replace raw compute masks and row-loop joins with Plan.hash_join.
- [ ] Ensure join-safe projections drop list payloads or explode prior to join.
- [ ] Record join precheck errors in finalize artifacts.

## Scope 06: Ordering + determinism enforcement for ingestion outputs

Intent: Make ordering transitions explicit, and enforce deterministic ordering at finalize.

Code pattern
```python
from codeintel.core.schemas.primitives import resolve_canonical_sort_keys
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table, FinalizeDedupe

order_by = resolve_canonical_sort_keys("core.syntax_nodes")
plan = plan.order_by(sort_keys=order_by)
spec = finalize_spec_for_table(
    "core.syntax_nodes",
    dedupe=FinalizeDedupe(enabled=True, tier="canonical"),
)
```

Targets
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/columnar/order_ing.py`
- `src/codeintel/ingestion/compute/base.py`
- `src/codeintel/build/hamilton/native/ingestion/manifesting.py`

Checklist
- [ ] Ensure ingestion plans declare ordering transitions (scan/join/aggregate/order_by).
- [ ] Require canonical tie-breakers for CANONICAL outputs.
- [ ] Apply stable sort + deterministic dedupe in finalize.

## Scope 07: Schema inference integration for ingestion outputs

Intent: Use plan schema inference for ingestion outputs and reduce output registry entries
to contract constraints (keys, ordering, determinism, extras policy).

Code pattern
```python
from codeintel.core.columnar.plan_schema import compile_plan_schema
from codeintel.core.schemas.service import get_schema_service

schema = compile_plan_schema(plan, inputs={"left": left.schema})
get_schema_service().register_plan_schema("core.syntax_nodes", schema)
```

Targets
- `src/codeintel/build/schemas/inference_service.py`
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/ingestion/compute/plan_surface.py`

Checklist
- [ ] Wire plan schema compiler into ingestion plan outputs.
- [ ] Reduce output registry entries to constraints only.
- [ ] Keep extras struct schemas aligned to inferred plan outputs.

## Scope 08: Provenance + observability artifacts for ingestion

Intent: Emit provenance columns when enabled and standardize error/alignment/stats artifacts
for ingestion runs, including list error tables.

Code pattern
```python
from codeintel.core.columnar.run_manifest import write_run_manifest
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

telemetry = scan_telemetry_for_queryspec(dataset, spec=spec)
write_run_manifest(
    manifest_dir=manifest_dir,
    context=ctx,
    scan_telemetry=telemetry,
    list_errors=list_errors_payload,
)
```

Targets
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/build/hamilton/native/ingestion/manifesting.py`
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/ingestion/scip/manifest.py`

Checklist
- [ ] Ensure provenance columns flow into ingestion error artifacts by default when enabled.
- [ ] Persist list error summaries and alignment stats in ingest manifests.
- [ ] Standardize run manifest payload fields for ingestion pipelines.

## Scope 09: Storage-free ingestion outputs (parquet-only)

Intent: Ensure ingestion does not persist into storage ports or DuckDB; outputs go to
parquet datasets and manifest-only metadata.

Code pattern
```python
from codeintel.build.exports.writers import write_parquet_dataset

write_parquet_dataset(
    table_key="core.syntax_nodes",
    reader=finalized_reader,
    output_dir=output_dir,
)
```

Targets
- `src/codeintel/ingestion/ports/*` (remove storage port usage)
- `src/codeintel/ingestion/adapters/*` (strip storage adapters)
- `src/codeintel/build/exports/writers.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`

Checklist
- [ ] Remove ingestion dependencies on storage/duckdb ports.
- [ ] Standardize parquet-only sinks for ingestion outputs.
- [ ] Keep run manifests as the only metadata persistence outside parquet.

## Rollout order (ingestion-specific)

1) Scope 01-03 (plan-first sources, QuerySpec standardization, reader-first finalize).
2) Scope 04-05 (kernel lane + join normalization for list-bearing ingestion joins).
3) Scope 06-08 (ordering/determinism + schema inference + observability artifacts).
4) Scope 09 (storage-free outputs cleanup).
