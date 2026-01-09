# Analytics Acero/DSL Intensification Plan

## Goal
Push analytics toward a plan-first, Arrow-acero-centric architecture with explicit
kernel lanes, reader-first execution, contract-driven determinism, and unified
observability artifacts. This is a design-phase plan and allows modular redesigns.

## Scope Items

### 1) Shared analytics pipeline runner (plan -> finalize)

Pattern
```python
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.plan_ops import build_query_plan_for_context
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec
from codeintel.build.analytics.utilities.finalize import finalize_analytics_reader


def run_analytics_pipeline(
    *,
    dataset: ds.Dataset,
    spec: QuerySpec,
    table_key: str,
    ctx: ExecutionContext,
) -> FinalizeResult:
    plan = build_query_plan_for_context(dataset, spec=spec, ctx=ctx)
    reader = plan.to_reader(use_threads=ctx.use_threads)
    return finalize_analytics_reader(table_key, reader)
```

Target files
- src/codeintel/build/analytics/utilities/pipeline.py (new)
- src/codeintel/build/analytics/utilities/finalize.py
- src/codeintel/build/analytics/utilities/datasets.py
- src/codeintel/build/analytics/utilities/snapshot.py

Checklist
- [x] Add a single runner that accepts QuerySpec + ExecutionContext and returns
      FinalizeResult.
- [x] Adopt the runner in post_run_quality_outputs (pilot path).
- [x] Add a pipeline request helper for datasets.
- [~] Remove ad hoc plan execution from remaining analytics modules and route
      through the runner.
- [~] Keep plan lane and kernel lane explicit (no compute in nodes); remaining
      modules still mix plan execution inline.

---

### 2) QuerySpec-driven snapshot/scan + provenance

Pattern
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec

projection = ProjectionSpec(
    base_cols=("repo", "commit", "function_goid_h128", "kind"),
    computed=(("kind_norm", E.cast(E.field("kind"), "string")),),
)
spec = QuerySpec(
    predicate=E.and_(
        E.field("repo") == E.scalar(repo),
        E.field("commit") == E.scalar(commit),
    ),
    pushdown_predicate=E.and_(
        E.field("repo") == E.scalar(repo),
        E.field("commit") == E.scalar(commit),
    ),
    projection=projection,
)
```

Target files
- src/codeintel/build/analytics/utilities/snapshot.py
- src/codeintel/build/analytics/utilities/datasets.py
- src/codeintel/build/analytics/utilities/catalogs.py
- src/codeintel/build/analytics/graphs/*
- src/codeintel/build/analytics/functions/*

Checklist
- [x] Replace custom snapshot filtering/projection with QuerySpec builders
      (entrypoints/core, data_models/core, semantic_roles/core,
      compute/dependencies, compute/data_models/usage, compute/functions/goids,
      functions/metrics).
- [~] Centralize provenance toggles in ExecutionContext and QuerySpec compilation
      (QuerySpec builders exist; table-based snapshots now accept ctx;
      post_run_quality_outputs uses ctx; remaining call sites still need
      adoption).
- [x] Ensure scan pushdown and plan filters share the same predicate semantics.

---

### 3) Rowset-first graph and AST pipelines

Pattern
```python
plan = snapshot_plan(table, repo=repo, commit=commit, columns=required)
plan = plan.filter(E.and_(E.is_valid("function_goid_h128"), E.is_valid("src_id")))
plan = plan.aggregate(
    keys=[E.field("function_goid_h128")],
    aggregates=[
        ("src_id", "list", None, "src_id"),
        ("dst_id", "list", None, "dst_id"),
        ("edge_kind", "list", None, "edge_kind"),
    ],
)
rowset = materialize_plan(plan, use_threads=True)

for row in iter_rows(rowset):
    src_ids = _list_values(row.get("src_id"))
    dst_ids = _list_values(row.get("dst_id"))
    # decode once at the graph boundary only
```

Target files
- src/codeintel/build/analytics/cfg_dfg/helpers.py
- src/codeintel/build/analytics/cfg_dfg/cfg_core.py
- src/codeintel/build/analytics/cfg_dfg/dfg_core.py
- src/codeintel/build/analytics/graphs/config_references.py
- src/codeintel/build/analytics/graphs/config_data_flow.py
- src/codeintel/build/analytics/subsystems/affinity.py

Checklist
- [x] Use Plan.aggregate(list) rowsets for adjacency inputs with order_by for list
      semantics.
- [~] Decode lists only at the final graph/AST boundary (cfg/dfg decode ordering
      added; more rowsets remain).
- [~] Keep ordering only when required for list semantics, not for determinism.

---

### 4) Columnar row builders for analytics outputs

Pattern
```python
from codeintel.core.columnar.rows import columnar_buffer_for_table_key

buffer = columnar_buffer_for_table_key("analytics.graph_metrics_functions")
for node in graph_nodes:
    buffer.append(
        {
            "repo": repo,
            "commit": commit,
            "function_goid_h128": int(node),
            "call_fan_in": len(in_neighbors.get(node, ())),
            "call_fan_out": len(out_neighbors.get(node, ())),
            "created_at": created_at,
        }
    )
table = buffer.to_table()
```

Target files
- src/codeintel/build/analytics/compute/row_builders/*
- src/codeintel/build/analytics/compute/graphs/*
- src/codeintel/build/analytics/compute/functions/*
- src/codeintel/build/analytics/compute/data_models/*
- src/codeintel/build/analytics/data_models/core.py
- src/codeintel/build/analytics/entrypoints/core.py

Checklist
- [ ] Replace list-of-dict row assembly with ColumnarRowBuffer builders.
- [ ] Align outputs with schema contracts before finalize.
- [ ] Avoid early table materialization unless needed at a boundary.

---

### 5) Determinism and ordering policy centralization

Pattern
```python
result = finalize_analytics_result(table_key, table)
return result.good
```

Target files
- src/codeintel/build/analytics/py_cpg_quality_report.py
- src/codeintel/build/analytics/scip_diagnostics_rollups.py
- src/codeintel/build/analytics/graphs/graph_metrics.py
- src/codeintel/build/analytics/graphs/graph_metrics_ext.py
- src/codeintel/build/analytics/graphs/subsystem_graph_metrics.py

Checklist
- [ ] Remove ad hoc order_by/sort in analytics modules used only for determinism.
- [ ] Use finalize_policy for canonical ordering + dedupe everywhere.
- [ ] Keep ordering only when list ordering changes semantics.

---

### 6) Guardrails for DSL compliance

Pattern
```python
if "pyarrow.compute" in text and "/analytics/" in path:
    raise SystemExit("Use DSL expr/kernels helpers instead of raw pc.")
```

Target files
- tools/lint_analytics_iter_rows.py
- tools/lint_analytics_rowset_guardrails.py
- tools/lint_no_raw_pyarrow_compute_in_nodes.py
- tools/lint_no_materialize_in_nodes.py
- tools/quality_report.py

Checklist
- [ ] Add allowlist-based guardrail for iter_rows in analytics modules.
- [ ] Disallow raw pc usage outside DSL helpers and rowset boundary code.
- [ ] Flag to_table/materialize_plan outside finalize boundaries for analytics.
- [ ] Wire guardrails into the quality report.

---

## Sequencing Recommendation
1) Shared pipeline runner + QuerySpec builders (Scopes 1-2).
2) Rowset-first graph pipelines + columnar row builders (Scopes 3-4).
3) Determinism centralization + guardrails (Scopes 5-6).
