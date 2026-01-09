# Core Compute Arrow Expansion Plan (Storage + Serving)

## Purpose
Extend the advanced, Arrow-centric compute architecture in storage and serving
while keeping DuckDB + SQLGlot as the primary planning backbone. This plan
targets the areas that still rely on materialized tables, Python loops, or
DuckDB-only processing, and brings them into the plan -> execute -> finalize
pipeline described in `docs/python_library_reference/compute_improvement_deepdive.md`.

## Guardrails
- DuckDB remains the canonical planner/engine for SQL + joins.
- Arrow/Acero is a downstream stage unless a query is explicitly Arrow-only.
- Arrow DSL subset is limited to filter/projection/order/limit initially.
- All outputs still pass through finalize gate for alignment + invariants.

---

## Scope Item 1: Arrow DSL subset translation for serving (filters/projections/order/limit)

### Rationale
Serving pipelines remain DuckDB-first, but can push a common subset of
query behavior into Arrow/Acero for plan-based streaming and deterministic
ordering without requiring full SQL execution outside DuckDB.

### Pattern to deploy
```python
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_ops import build_scan_plan
from codeintel.serving.semantic.arrow_plan_builder import ArrowPlanSpec

arrow_spec = ArrowPlanSpec(
    filter_expr=E.and_(E.field("repo") == E.scalar(repo), E.is_valid("commit")),
    projections={"repo": E.field("repo"), "commit": E.field("commit")},
    order_by=[("repo", "ascending"), ("commit", "ascending")],
    limit=100,
)

plan = build_scan_plan(
    dataset,
    columns=arrow_spec.projections,
    filter_expr=arrow_spec.filter_expr,
    order_by=arrow_spec.order_by,
)
reader = plan.to_reader(use_threads=use_threads)
```

### Target files
- src/codeintel/serving/semantic/arrow_plan_builder.py (new)
- src/codeintel/serving/semantic/query_ast.py
- src/codeintel/serving/semantic/sqlglot_query_builder.py
- src/codeintel/serving/semantic/duckdb_relation_builder.py

### Detailed checklist
- Introduce `ArrowPlanSpec` that captures filter/projection/order/limit.
- Add SQLGlot subset extraction that yields `ArrowPlanSpec` when safe.
- Keep SQLGlot canonicalization as the source of truth; only translate when
  expressions are representable in Arrow expressions.
- Route Arrow subset to `build_scan_plan` for pushdown and streaming readers.
- Fall back to DuckDB-only behavior when unsupported constructs appear.

---

## Scope Item 2: Arrow post-processing engine downstream of DuckDB

### Rationale
Keep DuckDB as the base engine, but allow an Arrow/Acero post-processing stage
to apply deterministic ordering, projections, or filters and then finalize in a
consistent Arrow boundary.

### Pattern to deploy
```python
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.conversion import reader_to_table

reader = relation.fetch_record_batch(batch_size)
table = reader_to_table(reader)

post = Plan.table(table).filter(filter_expr).project(projections).order_by(
    sort_keys=order_by,
)
post_reader = post.to_reader(use_threads=use_threads)

finalized = finalize_table(
    reader_to_table(post_reader),
    spec=FinalizeSpec(table_key=table_key, mode="tolerant"),
)
```

### Target files
- src/codeintel/serving/semantic/engines/arrow_engine.py (new)
- src/codeintel/serving/semantic/engines/registry.py
- src/codeintel/serving/semantic/engines/duckdb_engine.py
- src/codeintel/serving/semantic/duckdb_relation_builder.py

### Detailed checklist
- Add a new Arrow post-processing engine that consumes DuckDB relations.
- Only apply when `ArrowPlanSpec` is available and the query is non-join.
- Materialize once at a boundary, then use Acero `Plan.table` for post-ops.
- Preserve DuckDB as the fallback for joins or unsupported expressions.
- Keep finalize gate as the single correctness boundary after post-processing.

---

## Scope Item 3: Manifest parquet path rebuild (plan -> execute -> finalize)

### Rationale
Manifest tracking uses `dataset.to_table()` and `read_all()`, which bypasses
scan pushdown, provenance, and finalize-gate error artifacts.

### Pattern to deploy
```python
scan_options = DatasetScanOptions(
    columns=["target", "repo", "commit", "computed_at"],
    provenance_columns=("__filename", "__fragment_index", "__batch_index"),
    implicit_ordering=True,
    require_sequenced_output=True,
)
plan = build_scan_plan(
    dataset,
    columns=scan_options.projection_columns(),
    filter_expr=None,
)
reader = plan.to_reader(use_threads=use_threads)
finalized = finalize_table(reader_to_table(reader), spec=FinalizeSpec(...))
```

### Target files
- src/codeintel/storage/tracking/build_tracking.py
- src/codeintel/storage/datasets/manifest_index.py
- src/codeintel/core/columnar/streaming.py

### Detailed checklist
- Replace `dataset.to_table()` and `reader.read_all()` with `build_scan_plan`.
- Use scan control plane defaults (provenance, telemetry) for manifest reads.
- Finalize on read using `FinalizeSpec` and dedupe as part of finalize.
- Align deterministic ordering using explicit `order_by` or stable sort indices.

---

## Scope Item 4: Parquet query count and telemetry modernization

### Rationale
Some count paths still materialize tables. Use dataset preflight + plan
aggregations to avoid full table loads.

### Pattern to deploy
```python
plan = build_scan_plan(dataset, columns=["id"], filter_expr=filter_expr)
plan = plan.aggregate(keys=[], aggregates=[("id", "count", None, "row_count")])
row_count = reader_to_table(plan.to_reader())["row_count"][0].as_py()
```

### Target files
- src/codeintel/storage/queries/parquet.py
- src/codeintel/core/columnar/plan_ops.py

### Detailed checklist
- Replace `scanner.to_table()` / `dataset.to_table()` counts with plan aggregate.
- Use `get_fragments()` / `count_rows()` telemetry when available.
- Standardize `use_threads` and `implicit_ordering` for deterministic counts.

---

## Scope Item 5: Streaming finalize in repository and maintenance paths

### Rationale
Repository read helpers and maintenance scans still materialize tables
before finalization. Streaming finalize keeps batches flowing and reduces
memory overhead while preserving deterministic behavior.

### Pattern to deploy
```python
for batch in reader:
    result = finalize_table(
        pa.Table.from_batches([batch], schema=batch.schema),
        spec=FinalizeSpec(table_key=table_key, mode="tolerant"),
    )
    yield from records_from_arrow_table(result.good)
```

### Target files
- src/codeintel/storage/repositories/base.py
- src/codeintel/storage/datasets/maintenance.py
- src/codeintel/serving/export/ndjson.py

### Detailed checklist
- Add a streaming finalize helper that operates per batch.
- Replace `reader_to_table` in repository read paths when only row iteration is needed.
- For maintenance scans, finalize in streaming mode and only materialize where
  contract-driven aggregation is required.
- Preserve existing finalize artifacts for logging and diagnostics.

---

## Scope Item 6: Arrow-first paths for parquet-backed tracking/catalog tables

### Rationale
Tracking and catalog flows are still SQL-heavy even when parquet datasets are
available. Introduce Arrow scan + finalize paths and use DuckDB as fallback.

### Pattern to deploy
```python
if dataset_manifest is not None:
    dataset = dataset_for_entry(entry)
    plan = build_scan_plan(dataset, columns=columns, filter_expr=filter_expr)
    reader = plan.to_reader(use_threads=use_threads)
    table = finalize_table(reader_to_table(reader), spec=FinalizeSpec(...)).good
else:
    table = duckdb_relation.fetch_record_batch(batch_size).read_all()
```

### Target files
- src/codeintel/storage/tracking/run_tracking.py
- src/codeintel/storage/tracking/asset_tracking.py
- src/codeintel/storage/tracking/schema_catalog.py
- src/codeintel/storage/datasets/registry.py

### Detailed checklist
- Detect parquet-backed datasets and prefer Arrow scan pipelines.
- Apply `DatasetScanOptions` for consistent pushdown, provenance, telemetry.
- Keep DuckDB query paths as explicit fallbacks.
- Enforce finalize gate at the Arrow boundary for error artifacts + determinism.

---

## Optional follow-up: External runner integration (DataFusion)

### Rationale
Allow external engines for complex operations while keeping the finalize gate
as the single correctness boundary.

### Pattern to deploy
```python
from codeintel.core.columnar.plan_ops import ExternalPlanSpec, run_external_plan

reader = run_external_plan(
    ExternalPlanSpec(engine="datafusion", payload=substrait_bytes),
    dataset=dataset,
    filter_expr=filter_expr,
    columns=columns,
    scan_options=scan_options,
    use_threads=use_threads,
)
```

### Target files
- src/codeintel/core/columnar/plan_ops.py
- src/codeintel/serving/semantic/duckdb_relation_builder.py
- src/codeintel/serving/semantic/engines/registry.py

### Detailed checklist
- Implement a DataFusion runner behind `ExternalPlanSpec`.
- Register the runner in a single audited registry.
- Ensure all external outputs pass through finalize gate + ordering logic.
