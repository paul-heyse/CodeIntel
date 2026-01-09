# Ingestion Acero/DSL Maximal Framework Alignment Plan

## Objective
Reframe all ingestion behavior around the core Acero/DSL framework so every
ingestion output (parsing, tool runs, scans, joins, rollups) is expressed as
either a plan lane operation (scan/filter/project/join/aggregate) or a kernel
lane operation (explode, dedupe, ranking), with finalize as the only
materialization boundary.

## Non-Negotiables
- All ingestion scans use QuerySpec + runtime profiles.
- Plan lane only for Acero operations; kernel lane only for row-changing ops.
- Finalize is the only materialization boundary (reader-first everywhere).
- Schema policies define defaults (projection, join-safe columns, ordering).
- Provenance + telemetry are first-class and emitted on every finalize.

---

## Scope 01 — Ingestion DSL Facade + Runtime Profile Binding
**Goal**
Provide a single ingestion facade that constructs plans and executes them with
profile-driven defaults (threads, determinism, provenance), eliminating ad hoc
plan usage in ingestion nodes.

**Representative pattern**
```python
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.plan_ops import build_query_plan_for_context
from codeintel.ingestion.compute.queryspecs import build_ingest_query_spec

ctx = resolve_execution_context(None)
spec = build_ingest_query_spec(
    "core.modules",
    columns=("repo", "commit", "path", "module", "language"),
    repo=repo,
    commit=commit,
)
plan = build_query_plan_for_context(dataset, spec=spec, ctx=ctx)
reader = plan.to_reader(use_threads=ctx.runtime_profile.plan_use_threads)
```

**Target files**
- `src/codeintel/ingestion/compute/plan_surface.py`
- `src/codeintel/ingestion/compute/base.py`
- `src/codeintel/core/columnar/execution_context.py`
- `src/codeintel/core/columnar/plan_ops.py`

**Implementation checklist**
- [ ] Define a minimal ingestion facade: `ingest_plan_for_*` and
      `ingest_reader_for_*`.
- [ ] Bind ExecutionContext defaults (threads, determinism, provenance).
- [ ] Remove remaining ad hoc plan construction in ingestion nodes.

---

## Scope 02 — QuerySpec Control Plane for Ingestion Scans
**Goal**
Make QuerySpec the only scan surface for ingestion datasets, with schema-driven
defaults and consistent provenance inclusion.

**Representative pattern**
```python
from codeintel.core.columnar.plan_builder import SchemaPlanDefaultsRequest, plan_from_schema_defaults
from codeintel.core.schemas.service import get_schema_service

plan = plan_from_schema_defaults(
    schema_service=get_schema_service(),
    request=SchemaPlanDefaultsRequest(
        table_key="core.scip_occurrences",
        dataset=dataset,
        predicate=spec.predicate,
        columns=spec.projection.columns(),
        ctx=ctx,
    ),
)
reader = plan.to_reader(use_threads=ctx.runtime_profile.plan_use_threads)
```

**Target files**
- `src/codeintel/ingestion/compute/queryspecs.py`
- `src/codeintel/ingestion/compute/plan_surface.py`
- `src/codeintel/core/columnar/queryspec.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/streaming.py`

**Implementation checklist**
- [ ] Centralize ingest projection/predicate generation in QuerySpec helpers.
- [ ] Enforce a single scan builder (`build_query_plan_for_context`).
- [ ] Ensure provenance columns are included when determinism is canonical.

---

## Scope 03 — Kernel Lane Expansion for Ingestion
**Goal**
Standardize all row-changing operations and winner selection in kernel helpers
so ingestion never re-implements explode, rollups, or dedupe.

**Representative patterns**
```python
from codeintel.core.columnar.explode_ops import ExplodeSpec
from codeintel.core.columnar.plan_kernels import explode_edges_for_join

exploded = explode_edges_for_join(
    table=edges,
    spec=ExplodeSpec(
        src_col="src_id",
        dst_list_col="dst_ids",
        aligned_list_cols=("dst_spans",),
        repeat_cols=("repo", "commit", "rel_path"),
        null_list_policy="error",
    ),
    table_key="core.syntax_edges",
    schema_service=schema_service,
)
```

```python
from codeintel.core.columnar.plan_kernels import StableDedupeSpec, stable_dedupe_with_ties

deduped = stable_dedupe_with_ties(
    table,
    spec=StableDedupeSpec(
        key_columns=("repo", "commit", "scip_symbol"),
        order_by=(("match_priority", "descending"),),
        tie_breakers=(("def_rel_path", "ascending"), ("def_start_line", "ascending")),
    ),
)
```

**Target files**
- `src/codeintel/core/columnar/plan_kernels.py`
- `src/codeintel/core/columnar/explode_ops.py`
- `src/codeintel/core/columnar/dedupe_ops.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`

**Implementation checklist**
- [ ] Add a winner-selection helper (priority + tie-breakers).
- [ ] Replace ad hoc explode/list alignment with explode kernels.
- [ ] Route ingestion dedupe paths through kernel helpers only.

---

## Scope 04 — Ordering + Determinism Enforcement
**Goal**
Make ordering transitions explicit and enforce canonical ordering at finalize
boundaries for ingestion outputs.

**Representative pattern**
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.ordering import OrderingSpec

plan = ExecutionPlan.from_reader(reader, ordering=OrderingSpec.implicit(reason="ingest reader"))
result = run_pipeline(
    plan=plan,
    finalize=finalize_spec_for_table("core.syntax_nodes", mode="tolerant"),
    options=PipelineRunOptions(ctx=ctx),
)
```

**Target files**
- `src/codeintel/core/columnar/ordering.py`
- `src/codeintel/core/columnar/plan_ops.py`
- `src/codeintel/core/columnar/finalize_ops.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`

**Implementation checklist**
- [ ] Encode ordering transitions for join/aggregate/order_by in the plan layer.
- [ ] Require canonical tie-breakers for canonical determinism.
- [ ] Ensure finalize emits canonical ordering and artifacts consistently.

---

## Scope 05 — Schema-Driven Plan Defaults
**Goal**
Make schema metadata the sole authority for projection defaults and join-safe
allowlists used in ingestion plans.

**Representative pattern**
```python
from codeintel.core.schemas.primitives import PlanPolicy

TableSchema(
    ...,
    plan_policy=PlanPolicy(
        default_projection=("repo", "commit", "rel_path"),
        join_safe_columns=("repo", "commit", "rel_path"),
    ),
)
```

**Target files**
- `src/codeintel/core/schemas/primitives.py`
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/columnar/plan_builder.py`
- `src/codeintel/core/columnar/queryspec.py`

**Implementation checklist**
- [ ] Ensure all ingestion tables define PlanPolicy defaults.
- [ ] Replace call-site projection lists with schema-driven defaults.
- [ ] Use schema join-safe columns for all hash joins.

---

## Scope 06 — Provenance + Run Manifest Unification
**Goal**
Guarantee every ingestion finalize emits scan telemetry and a run manifest with
ordering, determinism tier, and profile metadata.

**Representative pattern**
```python
from codeintel.core.columnar.arrowdsl import PipelineRunOptions, run_pipeline
from codeintel.core.columnar.run_manifest import run_manifest_options_for_context
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec

telemetry = scan_telemetry_for_queryspec(dataset, spec=spec)
options = PipelineRunOptions(
    ctx=ctx,
    scan_telemetry=telemetry,
    manifest_dir=manifest_dir,
    manifest_options=run_manifest_options_for_context(
        ctx=ctx,
        ordering=plan.ordering,
        scan_telemetry=telemetry,
    ),
)
result = run_pipeline(plan=ExecutionPlan.from_plan(plan), finalize=finalize, options=options)
```

**Target files**
- `src/codeintel/core/columnar/run_manifest.py`
- `src/codeintel/core/columnar/streaming.py`
- `src/codeintel/build/hamilton/native/ingestion/manifesting.py`
- `src/codeintel/build/hamilton/transforms/ingestion_normalize.py`

**Implementation checklist**
- [ ] Require scan telemetry for all dataset scans.
- [ ] Emit run manifests for every ingestion finalize path.
- [ ] Include ordering/determinism/profile metadata in manifests.

---

## Scope 07 — Reader-First Boundaries + Guardrails
**Goal**
Prevent ad hoc materialization and raw compute sprawl in ingestion nodes.

**Representative pattern**
```python
from codeintel.core.columnar.plan_ops import materialize_plan

table = materialize_plan(plan, ctx=ctx)  # only at finalize or explicit debug boundary
```

**Target files**
- `tools/lint_no_materialize_in_nodes.py` (new)
- `tools/lint_no_raw_pyarrow_compute_in_nodes.py` (extend)
- `src/codeintel/build/hamilton/native/ingestion/*`
- `src/codeintel/ingestion/compute/*`

**Implementation checklist**
- [ ] Add guardrail lint for `.to_table()`/`read_all()` outside finalize.
- [ ] Extend raw pyarrow compute lint to ingestion modules.
- [ ] Gate ingestion nodes to reader outputs only.

---

## Scope 08 — Compute Modules: Parser Outputs to Readers Only
**Goal**
Keep AST/CST/tree-sitter/symtable/dis/docstrings/inspect steps in kernel lane,
but emit readers only and rely on plan + finalize downstream.

**Representative pattern**
```python
from codeintel.core.columnar.rows import columnar_batch_collector_for_table_key

collector = columnar_batch_collector_for_table_key("core.syntax_nodes", batch_size=4096)
collector.append({"repo": repo, "commit": commit, "node_id": node_id, "kind": kind})
reader = collector.to_reader()
```

**Target files**
- `src/codeintel/ingestion/compute/ast_extract.py`
- `src/codeintel/ingestion/compute/cst_extract.py`
- `src/codeintel/ingestion/compute/tree_sitter_index.py`
- `src/codeintel/ingestion/compute/symtable_extract.py`
- `src/codeintel/ingestion/compute/dis_extract.py`
- `src/codeintel/ingestion/compute/docstrings_extract.py`
- `src/codeintel/ingestion/compute/inspect_extract.py`

**Implementation checklist**
- [ ] Remove any `to_table()` calls from compute modules.
- [ ] Ensure each module returns readers plus row counts.
- [ ] Shift filtering/dedupe/ordering to plan kernels downstream.

---

## Scope 09 — Tool Ingestion Unification
**Goal**
Normalize tool outputs into Arrow readers and process them through the DSL
pipeline to enforce consistent ordering, dedupe, and observability.

**Representative pattern**
```python
from codeintel.core.columnar.rows import columnar_buffer_for_table_key

buffer = columnar_buffer_for_table_key("analytics.static_diagnostics")
buffer.append({"repo": repo, "commit": commit, "path": path, "count": count})
reader = buffer.to_reader()
```

**Target files**
- `src/codeintel/ingestion/compute/typing_ingest.py`
- `src/codeintel/ingestion/compute/tests_ingest.py`
- `src/codeintel/ingestion/compute/config_ingest.py`
- `src/codeintel/ingestion/engine/*`

**Implementation checklist**
- [ ] Convert tool outputs into columnar readers immediately.
- [ ] Pass tool readers through DSL finalize wrappers.
- [ ] Emit tool run telemetry in manifests (extras payloads).

---

## Scope 10 — Repo Scan + Change Detection Alignment
**Goal**
Keep scanning/change detection in kernel lane but move all post-processing into
plan + finalize, with dataset-first persistence (Parquet only).

**Representative pattern**
```python
from codeintel.core.columnar.rows import columnar_buffer_for_table_key

buffer = columnar_buffer_for_table_key("core.modules")
buffer.append({"repo": repo, "commit": commit, "path": path, "language": language})
reader = buffer.to_reader()
```

**Target files**
- `src/codeintel/ingestion/compute/repo_scan.py`
- `src/codeintel/ingestion/ports/change_detection.py`
- `src/codeintel/ingestion/adapters/hash_change_detection.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`

**Implementation checklist**
- [ ] Remove any storage port usage from ingestion paths.
- [ ] Ensure scan outputs are readers only and finalized centrally.
- [ ] Persist via dataset/Parquet-only flows.

---

## Sequencing Recommendation
1) Scope 01 + 02 (DSL facade + QuerySpec control plane)
2) Scope 03 + 04 (kernel lane + ordering/determinism enforcement)
3) Scope 05 + 06 (schema defaults + manifest/telemetry)
4) Scope 07 + 08 + 09 + 10 (guardrails + compute/tool/scan migration)

## Validation (Non-Pytest)
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Validate run manifests include ordering + determinism + scan telemetry for
  representative ingestion targets.
