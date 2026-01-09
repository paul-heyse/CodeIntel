# Core Columnar Canonical Protocol Unification Plan

## Objective
Eliminate duplicate compute protocols around Arrow Acero/DSL by enforcing a small
set of canonical core surfaces in `src/codeintel/core`. Downstream layers may
wrap or re-export, but they should not re-implement the same computation shapes.

## Guiding Constraints (from the Arrow Acero/DSL guide)
- QuerySpec is the single query control plane for scan + plan.
- Plan lane (Acero) is the only place for scan/filter/project/join/aggregate.
- Kernel lane is the only place for row-changing ops (dedupe, explode, rollup).
- Finalize is the only materialization boundary and owns determinism.
- Schema policy is the single source of truth for projection, join-safety, and
  ordering.

---

## Scope 01 - QuerySpec as the Single Control Plane
**Goal**  
Remove alternate query definitions (for example, QueryPlanSpec) and make
`QuerySpec` the only surface that defines scan filters, projections, and Acero
plan defaults.

**Status**  
Completed.

**Code pattern**
```python
from codeintel.core.columnar.plan_builder import SchemaPlanDefaultsRequest, plan_from_schema_defaults
from codeintel.core.columnar.queryspec import QuerySpec
from codeintel.core.schemas.service import get_schema_service

spec = QuerySpec(
    table_key="analytics.function_metrics",
    columns=("repo", "commit", "function_goid_h128"),
    predicate=predicate,
)

plan = plan_from_schema_defaults(
    schema_service=get_schema_service(),
    request=SchemaPlanDefaultsRequest(
        table_key=spec.table_key,
        dataset=dataset,
        predicate=spec.predicate,
        columns=spec.scan_columns(provenance=False),
        ctx=execution_ctx,
    ),
)
```

**Target files**
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/streaming.py`

**Implementation checklist**
- [x] Remove QueryPlanSpec and any parallel query definition objects.
- [x] Make all dataset scans accept only QuerySpec and compile it for scan + plan.
- [x] Ensure provenance columns are derived via QuerySpec (not ad hoc lists).
- [x] Add a single helper to translate QuerySpec to scan telemetry input.

---

## Scope 02 - Kernel Lane Consolidation
**Goal**  
Ensure row-changing operations live only in `kernels.py` and
`plan_kernels.py`, and eliminate duplicate helpers elsewhere in core.

**Status**  
Completed.

**Code pattern**
```python
from codeintel.core.columnar.plan_kernels import (
    GroupedRollupSpec,
    StableDedupeSpec,
    explode_edges_for_join,
    grouped_rollup_table,
    stable_dedupe_with_ties,
)

rolled = grouped_rollup_table(
    table,
    spec=GroupedRollupSpec(
        keys=("repo", "commit"),
        aggregates=(("severity", "count", None, "diagnostic_count"),),
    ),
    ctx=execution_ctx,
)

deduped = stable_dedupe_with_ties(
    table,
    spec=StableDedupeSpec(
        key_columns=("repo", "commit", "function_goid_h128"),
        order_by=(("score", "descending"),),
        tie_breakers=(("rel_path", "ascending"),),
        require_tie_breakers=True,
    ),
)

exploded = explode_edges_for_join(
    table,
    spec=ExplodeSpec(src_col="src_id", dst_list_col="dst_ids"),
    table_key="core.syntax_edges",
    schema_service=schema_service,
)
```

**Target files**
- `src/codeintel/core/columnar/plan_kernels.py`
- `src/codeintel/core/columnar/kernels.py`
- `src/codeintel/core/columnar/groupby.py`
- `src/codeintel/core/columnar/compute.py`

**Implementation checklist**
- [x] Move group-by and aggregate helpers into `kernels.py` or `plan_kernels.py`.
- [x] Remove duplicate helpers that wrap pyarrow.compute outside kernel lane.
- [x] Ensure downstream modules re-export kernel helpers instead of redefining.
- [x] Keep all new row-changing logic in kernel lane only.

---

## Scope 03 - Schema as the Single Source of Truth
**Goal**  
Make schema metadata authoritative for projection defaults, join-safe columns,
and finalize policies. Prevent parallel policy maps or duplicated schema logic.

**Status**  
Completed.

**Code pattern**
```python
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.schemas.service import get_schema_service

schema_service = get_schema_service()
schema = schema_service.require_table_schema("core.modules")

finalize = finalize_spec_for_table(schema.table_key, mode="tolerant")
plan_policy = schema.plan_policy
```

**Target files**
- `src/codeintel/core/schemas/service.py`
- `src/codeintel/core/schemas/primitives.py`
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/columnar/join_safe.py`

**Implementation checklist**
- [x] Derive plan policy and finalize policy from the schema only.
- [x] Remove ad hoc policy maps and fallback policy logic.
- [x] Ensure TableSchema is populated by the canonical provider and cached in SchemaService.
- [x] Keep output_registry overrides only for truly non-inferable schemas.

---

## Scope 04 - Streaming Adapter Unification
**Goal**  
Make `ColumnarStream` the only conversion protocol for reader/table/lazyframe
and keep streaming utilities focused on dataset scanning and telemetry.

**Code pattern**
```python
from codeintel.core.columnar.stream import ColumnarStream, RecordBatchReaderStream

def stream_from_reader(reader: pa.RecordBatchReader) -> ColumnarStream:
    return RecordBatchReaderStream(reader)
```

**Target files**
- `src/codeintel/core/columnar/stream.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/core/columnar/readers.py`
- `src/codeintel/core/columnar/conversion.py`

**Implementation checklist**
- [ ] Standardize conversions via ColumnarStream adapters.
- [ ] Remove duplicate conversion helpers or make them thin wrappers.
- [ ] Keep dataset scan logic and telemetry in streaming.py only.
- [ ] Ensure downstream surfaces consume ColumnarStream rather than raw readers.

---

## Scope 05 - Runtime Profiles Only via ExecutionContext
**Goal**  
Make ExecutionContext the single runtime policy carrier for threading and
scan profiles. Remove parallel profile logic in helper modules.

**Code pattern**
```python
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.profiles import RuntimeProfile
from codeintel.core.columnar.streaming import configure_arrow_threading_for_context

ctx = ExecutionContext(runtime_profile=RuntimeProfile.DEV_FAST)
configure_arrow_threading_for_context(ctx=ctx)
```

**Target files**
- `src/codeintel/core/columnar/execution_context.py`
- `src/codeintel/core/columnar/profiles.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/core/columnar/runtime.py`

**Implementation checklist**
- [ ] Route all scan/plan threading decisions through ExecutionContext.
- [ ] Remove local runtime knobs that bypass ExecutionContext.
- [ ] Keep runtime profile defaults in one module (profiles.py).
- [ ] Ensure scan profile selection is derived from ExecutionContext only.

---

## Scope 06 - Finalize Policy Derivation (Schema-Only)
**Goal**  
Ensure finalize policies and determinism are derived only from schema metadata,
never recomputed at call sites.

**Code pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table

finalize = finalize_spec_for_table("analytics.function_metrics", mode="tolerant")
result = run_pipeline(
    plan=ExecutionPlan.from_plan(plan),
    finalize=finalize,
    options=PipelineRunOptions(ctx=execution_ctx),
)
table = result.good
```

**Target files**
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/core/columnar/ordering.py`
- `src/codeintel/core/schemas/primitives.py`

**Implementation checklist**
- [ ] Derive canonical ordering and dedupe specs strictly from schema policy.
- [ ] Remove any call-site logic that recomputes required_non_null or dedupe keys.
- [ ] Enforce stable_sort_keys precedence when present in schema policy.

---

## Sequencing (Recommended)
1) Scope 01 (QuerySpec single control plane) — completed.  
2) Scope 03 (Schema single source of truth) — completed.  
3) Scope 06 (Finalize policy derivation from schema).  
4) Scope 02 (Kernel lane consolidation).  
5) Scope 04 (Streaming adapter unification).  
6) Scope 05 (Runtime profiles via ExecutionContext).
4) Scope 02 (Kernel lane consolidation).  
5) Scope 04 (Streaming adapter unification).  
6) Scope 05 (Runtime profiles via ExecutionContext).

## Validation Gates (when re-enabled)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted unit tests for core columnar and schema modules.
