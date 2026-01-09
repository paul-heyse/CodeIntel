# Core Compute Acero DSL Maximal Adoption Extension Plan

## Goal
Extend the Arrow Acero + DSL adoption across the remaining build/ingestion surfaces so
all compute paths converge on "plan -> execute -> finalize" and become schema-inference
friendly without runtime execution.

## Alignment
This plan is an extension to `plans/core_compute_acero_dsl_maximal_adoption_plan.md`.
It focuses on additional targets not yet listed there, using the same shared DSL/Acero
building blocks (Plan, QuerySpec, ExecutionContext, kernel lane, finalize gate).

## Guiding Principles
- Prefer `codeintel.core.columnar.*` DSL and helpers as the canonical surface.
- Replace row-wise Python loops and `from_pylist` tables with Plan + kernel lane helpers.
- Keep readers streaming until finalize boundaries (`finalize_reader` when possible).
- Encode determinism and invariants in FinalizeSpec, not in ad-hoc node logic.

---

## Scope Items (Extension Targets)

### 1) Graph relation builders: migrate Python row loops to Plan pipelines
Pattern (QuerySpec + Plan scan + joins + finalize)
```python
from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, run_pipeline
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.finalize_ops import FinalizeSpec
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan, QuerySpec

spec = QuerySpec(
    predicate=E.and_(E.is_valid("goid_h128"), E.is_valid("rel_path")),
    pushdown_predicate=E.is_valid("rel_path"),
    projection=ProjectionSpec(base_cols=("repo", "commit", "rel_path", "goid_h128")),
)
plan = build_query_plan(dataset, spec=spec, provenance=False)
plan = plan.hash_join(right=other_plan, spec=HashJoinSpec(left_keys=["rel_path"], right_keys=["path"]))
result = run_pipeline(
    plan=ExecutionPlan(inner=plan.declaration),
    finalize=FinalizeSpec(table_key="graph.call_graph_edges", mode="tolerant"),
    ctx=ExecutionContext(determinism="canonical", provenance=False),
)
```

Target files
- src/codeintel/build/hamilton/native/graphs/call_graph.py
- src/codeintel/build/hamilton/native/graphs/goids.py
- src/codeintel/build/hamilton/native/graphs/import_graph.py
- src/codeintel/build/hamilton/native/graphs/pdg.py
- src/codeintel/build/hamilton/native/graphs/symbol_use.py
- src/codeintel/build/graphs/compute/goid.py

Checklist
- [x] Replace row-wise filtering (`iter_rows`) with Plan.filter + compute masks.
- [x] Replace `pa.Table.from_pylist` assemblies with Plan projections or kernel helpers.
- [ ] Add deterministic order_by via FinalizeSpec for contract outputs.
- [ ] Use `finalize_reader` when outputs are immediately materialized.

Status
- In progress (Phase 1 complete for `src/codeintel/build/hamilton/native/graphs/call_graph.py`, `src/codeintel/build/hamilton/native/graphs/goids.py`, `src/codeintel/build/hamilton/native/graphs/import_graph.py`, `src/codeintel/build/hamilton/native/graphs/symbol_use.py`; remaining: `src/codeintel/build/hamilton/native/graphs/pdg.py`, `src/codeintel/build/graphs/compute/goid.py`).

---

### 2) CPG2 assembly helpers: replace manual list assembly with kernel lane + plan
Pattern (explode + project + finalize)
```python
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.finalize_ops import FinalizeSpec
from codeintel.core.columnar.plan_ops import Plan

exploded = explode_edges(parent_table, spec=ExplodeSpec(src_col="edge_id", dst_list_col="edge"))
plan = Plan.table(exploded.good).project({name: E.field(("edge", name)) for name in edge_fields})
result = run_pipeline(
    plan=ExecutionPlan(inner=plan.declaration),
    finalize=FinalizeSpec(table_key="graph.cpg_edges", mode="strict"),
)
```

Target files
- src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py
- src/codeintel/build/hamilton/native/graphs/cpg2/ids.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py

Checklist
- [x] Replace `pa.Table.from_pylist` edge builds with explode + Plan projection.
- [ ] Enforce join-safe schemas before hash joins.
- [x] Route outputs through FinalizeSpec with canonical sort keys.

Status
- Complete for `src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py`, `src/codeintel/build/hamilton/native/graphs/cpg2/ids.py`, `src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py`, `src/codeintel/build/hamilton/native/graphs/cpg2/planes/link.py`, `src/codeintel/build/hamilton/native/graphs/cpg2/planes/symbol.py`.

---

### 3) Hamilton ingestion pipelines: move joins to Plan + join-precheck + finalize
Pattern (precheck + hashjoin + finalize_reader)
```python
from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, precheck_join_keys, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeSpec
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan

precheck = precheck_join_keys(left_table, spec=JoinPrecheckSpec(required_non_null=join_keys))
plan = Plan.table(precheck.good).hash_join(right=Plan.table(right_table), spec=HashJoinSpec(...))
result = run_pipeline(
    plan=ExecutionPlan(inner=plan.declaration),
    finalize=FinalizeSpec(table_key="core.syntax_xref", mode="tolerant"),
    ctx=ExecutionContext(determinism="canonical"),
)
```

Target files
- src/codeintel/build/hamilton/native/ingestion/syntax_augment.py
- src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py
- src/codeintel/build/hamilton/native/ingestion/scip.py
- src/codeintel/build/hamilton/native/ingestion/scip_resolution.py
- src/codeintel/build/hamilton/native/ingestion/scip_proto.py
- src/codeintel/build/hamilton/native/ingestion/tree_sitter.py
- src/codeintel/build/hamilton/native/ingestion/extraction_targets.py
- src/codeintel/build/hamilton/native/ingestion/ingest_targets.py
- src/codeintel/build/hamilton/native/ingestion/file_line_index.py
- src/codeintel/build/hamilton/native/ingestion/frame_utils.py
- src/codeintel/build/hamilton/native/ingestion/pipelines.py

Checklist
- [ ] Replace ad-hoc joins with HashJoinSpec + precheck_join_keys.
- [ ] Move join filtering to Plan.filter nodes for deterministic inference.
- [ ] Replace table materialization with finalize_reader where possible.

---

### 4) Ingestion compute steps (non-Hamilton): align to finalize + typed extras
Pattern (ColumnarRowBuffer + finalize_ingest_table)
```python
from codeintel.core.columnar.rows import columnar_buffer_for_table_key, table_for_columnar_rows
from codeintel.core.columnar.finalize_ops import finalize_table, FinalizeSpec

buffer = columnar_buffer_for_table_key("core.typing_diagnostics")
buffer.append(row_dict)
table = table_for_columnar_rows(buffer, table_key="core.typing_diagnostics")
result = finalize_table(table, spec=FinalizeSpec(table_key="core.typing_diagnostics", mode="tolerant"))
```

Target files
- src/codeintel/ingestion/compute/config_ingest.py
- src/codeintel/ingestion/compute/typing_ingest.py
- src/codeintel/ingestion/compute/tests_ingest.py
- src/codeintel/ingestion/compute/docstrings_extract.py

Checklist
- [ ] Replace dict-list assembly with ColumnarRowBuffer.
- [ ] Enforce typed `extras` struct via shared helpers where present.
- [ ] Route outputs through finalize_ingest_table for alignment + artifacts.

---

### 5) Analytics pipelines (native + compute): Plan-first aggregates + finalize
Pattern (aggregate + order_by + finalize)
```python
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeSpec

plan = Plan.table(input_table).aggregate(
    keys=[E.field("repo"), E.field("commit")],
    aggregates=[("goid_h128", "count", None, "goid_count")],
).order_by(sort_keys=[("repo", "ascending"), ("commit", "ascending")])
result = run_pipeline(
    plan=ExecutionPlan(inner=plan.declaration),
    finalize=FinalizeSpec(table_key="analytics.graph_metrics", mode="tolerant"),
)
```

Target files
- src/codeintel/build/hamilton/native/analytics/*.py
- src/codeintel/build/analytics/graphs/*.py
- src/codeintel/build/analytics/functions/*.py
- src/codeintel/build/analytics/subsystems/*.py
- src/codeintel/build/analytics/compute/*.py
- src/codeintel/build/analytics/entrypoints/*.py
- src/codeintel/build/analytics/semantic_roles/*.py
- src/codeintel/build/analytics/py_cpg_quality_report.py

Checklist
- [ ] Replace `iter_rows` loops with Plan.aggregate/group_by kernels.
- [ ] Use Plan.order_by + FinalizeSpec for deterministic analytics outputs.
- [ ] Remove `pa.Table.from_pylist` table builds in favor of Plan + kernels.

---

### 6) Graph engine + validation: reader-first scans and plan-level checks
Pattern (QuerySpec + reader-first finalize)
```python
from codeintel.core.columnar.plan_ops import build_query_plan
from codeintel.core.columnar.finalize_ops import finalize_reader, FinalizeSpec

plan = build_query_plan(dataset, spec=spec, provenance=True)
reader = plan.to_reader(use_threads=True)
finalized = finalize_reader(reader, spec=FinalizeSpec(table_key="graph.call_graph_edges", mode="tolerant"))
```

Target files
- src/codeintel/build/graphs/engine/datasets.py
- src/codeintel/build/graphs/engine/views.py
- src/codeintel/build/graphs/validation/checks/*.py
- src/codeintel/build/graphs/validation/runner.py

Checklist
- [ ] Replace materialized table scans with Plan scans + reader streaming.
- [ ] Push filters/projections into QuerySpec for deterministic semantics.
- [ ] Attach scan telemetry via shared scan helpers.

---

### 7) Exports + materializers: unify projection + finalize before write
Pattern (QuerySpec + finalize + write)
```python
from codeintel.core.columnar.plan_ops import build_query_plan
from codeintel.core.columnar.finalize_ops import finalize_table, FinalizeSpec

plan = build_query_plan(dataset, spec=spec, provenance=False)
table = plan.to_table(use_threads=True)
finalized = finalize_table(table, spec=FinalizeSpec(table_key=table_key, mode="strict"))
writer.write_table(table_key, finalized.good)
```

Target files
- src/codeintel/build/exports/common.py
- src/codeintel/build/exports/engine.py
- src/codeintel/build/exports/validation.py
- src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py
- src/codeintel/build/hamilton/materializers/arrow_parquet_cache.py

Checklist
- [ ] Apply QuerySpec for projection/filter consistency in export paths.
- [ ] Finalize outputs prior to writing artifacts.
- [ ] Preserve canonical ordering for cached/exported datasets.

---

### 8) Compute filters -> DSL expressions for inference friendliness
Pattern (expr_vocab expressions)
```python
from codeintel.core.columnar.expr_vocab import E

expr = E.and_(E.is_valid("rel_path"), E.eq("language", "python"))
plan = plan.filter(expr)
```

Target files
- src/codeintel/build/hamilton/native/graphs/compute_filters.py
- src/codeintel/build/tabular/compute_masks.py

Checklist
- [ ] Re-export filter expressions from core expr vocab.
- [ ] Remove table-level filter helpers that directly call pc kernels.
- [ ] Keep mask helpers in core columnar module for reuse.

---

### 9) Replace `from_pylist` and manual row assembly with columnar builders
Pattern (ColumnarRowBuffer / ColumnarBatchCollector)
```python
from codeintel.core.columnar.rows import ColumnarRowBuffer, table_for_columnar_rows

buffer = ColumnarRowBuffer()
for row in rows:
    buffer.append(row)
output = table_for_columnar_rows(buffer, table_key=table_key)
```

Target files
- src/codeintel/build/analytics/utilities/datasets.py
- src/codeintel/build/hamilton/native/graphs/goids.py
- src/codeintel/build/hamilton/native/graphs/cpg2/edge_helpers.py
- src/codeintel/build/hamilton/native/graphs/cpg2/planes/scip.py
- src/codeintel/build/hamilton/native/ingestion/syntax_augment.py

Checklist
- [ ] Remove `pa.Table.from_pylist` usage in build/ingestion nodes.
- [ ] Standardize row assembly through ColumnarRowBuffer or batch collectors.
- [ ] Route output through finalize gate after assembly.

---

### 10) Streaming safety and zero-copy iteration (beyond current plan)
Pattern (iter_array_values / iter_rows)
```python
from codeintel.core.columnar.iter import iter_array_values

values = list(iter_array_values(table["count"]))
```

Target files
- src/codeintel/build/tabular/arrow_ops.py
- src/codeintel/build/analytics/utilities/datasets.py
- src/codeintel/build/hamilton/native/graphs/call_graph.py
- src/codeintel/build/graphs/validation/checks/database.py

Checklist
- [ ] Replace `to_numpy`/`to_pylist` with streaming iter helpers.
- [ ] Keep readers unmaterialized until finalize boundaries.
- [ ] Add tiny tests for iter helpers where the behavior differs.

---

## Sequencing Recommendation (Extension)
1) Graph relation builders + CPG2 assembly (scope items 1-2).
2) Hamilton ingestion pipelines (item 3) + ingestion compute steps (item 4).
3) Analytics pipelines (item 5).
4) Graph engine + validation (item 6).
5) Exports/materializers + streaming safety + filter DSL (items 7-10).

## Expected Outcome
After this extension, the majority of build/ingestion compute will be expressed as
Acero plans with shared DSL helpers, making static, inference-driven schema derivation
practical without runtime execution or large manual declarations.
