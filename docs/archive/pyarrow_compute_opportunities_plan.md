# PyArrow Compute Expansion Plan (Storage Queries + Graph Prefiltering)

## Overview
This plan extends PyArrow compute usage beyond the prior build-graph refactors to
cover storage query validation and graph prefiltering hot paths. The goal is to
reduce Python row loops, improve pushdown/columnar execution, and keep behavior
stable with deterministic fallbacks when compute kernels are unsupported.

Primary references:
- `docs/python_library_reference/pyarrow-advanced.md`
- `docs/python_library_reference/pyrarrow.md`

## Goals
- Replace Python-level row loops in dataset validation queries with Arrow compute.
- Push down filters into Arrow tables before AST/graph logic executes in Python.
- Keep fallbacks for kernel or type limitations while preserving current behavior.
- Keep all touched files clean under Ruff, Pyright strict, and Pyrefly.

## Non-Goals
- Rewriting AST/graph construction to pure compute.
- Introducing experimental UDFs unless a concrete, measurable benefit appears.
- Changing table schemas or external contracts.

## Scope Summary (Target Files)
- Storage query validation: `src/codeintel/storage/queries/parquet.py`
- Arrow ops dedupe fallback: `src/codeintel/build/tabular/arrow_ops.py`
- Graph prefiltering:
  - `src/codeintel/build/hamilton/native/graphs/call_graph.py`
  - `src/codeintel/build/hamilton/native/graphs/import_graph.py`
  - `src/codeintel/build/hamilton/native/graphs/goids.py`
  - `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
  - `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
- Constant column injection:
  - `src/codeintel/build/hamilton/native/graphs/pdg.py`
- Shared helpers:
  - `src/codeintel/build/tabular/compute_helpers.py`
  - `src/codeintel/build/tabular/compute_masks.py`
  - `src/codeintel/build/hamilton/native/graphs/compute_filters.py`

## Status Summary (Current)
- Workstream A: Completed (A1-A3 done; A4 deferred).
- Workstream B: Completed (B1 complete; `is_in_mask` now supports sequence value sets).
- Workstream C: Completed.
- Workstream D: Completed.
- Workstream E: Partial (E1 covered by existing tests, E2 partial, E3 done, E4 targeted checks done; full guardrails/pytest blocked by Hamilton runtime type mismatch).

## Assumptions
- Dataset columns are Arrow-native types in most storage-query cases.
- Graph pipelines still require Python-level AST parsing; compute is for prefiltering.
- Existing behavior must be preserved; compute is a fast path, not a semantic change.

## Workstream A: Storage Query Compute Refactor
Target: `src/codeintel/storage/queries/parquet.py`

### A1. `safe_count_non_positive` compute-first path
Checklist:
- Detect numeric columns (`pa.types.is_integer`, `pa.types.is_floating`, `pa.types.is_decimal`).
- For numeric types, build `less_equal` mask via `pc.call_function("less_equal", ...)`.
- Count matches with `pc.call_function("sum", [mask])` and coerce to `int`.
- Preserve existing Python fallback for non-numeric or compute failure.

Acceptance:
- Numeric columns use compute; non-numeric fall back to existing behavior.
- Returned count matches current logic for mixed nulls/booleans.

Status: Completed (compute-first path implemented with safe fallbacks).

### A2. `safe_count_duplicates` compute path
Checklist:
- Use `pc.call_function("count", ...)` with `ScalarAggregateOptions(skip_nulls=True)` to get
  non-null count.
- Use `pc.call_function("count_distinct", ...)` to get distinct count.
- Compute duplicates = non-null count - distinct count.
- Keep existing Python fallback for unsupported types or kernel errors.

Acceptance:
- Duplicate counts match prior output for numeric, string, and dictionary-encoded columns.

Status: Completed (compute path implemented with safe fallbacks).

### A3. `safe_count_orphan_refs` compute path
Checklist:
- Normalize source/target to `pa.Array`/`pa.ChunkedArray` and prefilter nulls in target.
- Prefer compute set lookup: `pc.call_function("is_in", [source, target], options=...)`.
- Build orphan mask via `pc.call_function("invert", ...)` and `pc.call_function("is_null", ...)`.
- Apply `allow_null` rules with `and_kleene`/`or_kleene`.
- Count orphans with `pc.call_function("sum", [mask])`.
- Keep Python fallback when kernels are unsupported.

Acceptance:
- Orphan counts match current behavior for allow-null True/False.

Status: Completed (compute path implemented with safe fallbacks).

### A4. Optional helper extraction
Checklist:
- Introduce small helpers in `compute_helpers` for:
  - `safe_scalar(name, args, options)` returning Python scalar or None.
  - `safe_array(name, args)` returning Arrow array or None.
- Use helpers in `parquet.py` to reduce duplicated try/except blocks.

Acceptance:
- Helper usage in at least two call sites without changing behavior.

Status: Deferred (existing local helpers in `parquet.py` already handle failures).

## Workstream B: Graph Prefiltering With Compute
Targets: `call_graph.py`, `import_graph.py`, `goids.py`, `cfg_dfg.py`, `symbol_use.py`

### B1. Extend compute masks (if needed)
Checklist:
- Add `in_list_mask(values, allowed: Sequence[str])` to `compute_masks`.
- Add `node_type_is_function(values)` or similar for AST node-type filtering.
- Ensure masks handle nulls via `and_kleene`/`or_kleene`.

Acceptance:
- Masks are unit-tested with mixed nulls and unexpected types.

Status: Completed for `node_type_is_function`; `is_in_mask` now supports sequence value sets.

### B2. `call_graph.py` prefilter `core.modules` and `core.goids`
Checklist:
- Use `filter_python_modules` before `to_pylist()` (already available).
- For goids, use `filter_python_goids` before iterating rows.
- Keep AST parsing unchanged; only reduce input rows.

Acceptance:
- Output graph rows unchanged for existing tests.

Status: Completed (filters already in place).

### B3. `import_graph.py` module filter
Checklist:
- Apply `filter_python_modules` to `modules_table` before iteration.
- Keep language gate in Python for AST parsing.

Acceptance:
- No behavioral change; fewer rows hit Python.

Status: Completed (filters already in place).

### B4. `goids.py` module frame filter
Checklist:
- Use `filter_modules_with_language` before `_module_frame` iteration.
- Keep row shaping logic intact.

Acceptance:
- Module row list is identical after filtering for valid inputs.

Status: Completed (filters already in place).

### B5. `cfg_dfg.py` goid/ast prefilter
Checklist:
- Filter `goids_table` using `filter_python_goids` before `to_pylist()`.
- Add an AST node filter to exclude non-function entries early (compute mask on `node_type`).

Acceptance:
- CFG/DFG outputs unchanged for existing fixtures.

Status: Completed (compute prefilter added for AST function nodes and goids).

### B6. `symbol_use.py` occurrences/goids prefilter
Checklist:
- Apply `filter_symbol_occurrences` to occurrences before `to_pylist()`.
- Apply `filter_goids_with_spans` to goids before `to_pylist()`.

Acceptance:
- Symbol use edges unchanged; fewer rows processed in Python.

Status: Completed (filters already in place).

## Workstream C: Constant Column Injection
Target: `src/codeintel/build/hamilton/native/graphs/pdg.py`

### C1. Replace manual append loops with compute helpers
Checklist:
- Replace `pa.array(["DFG"] * n)` with `constant_array("DFG", n)`.
- Replace repeated `pa.nulls(n)` with `append_constant_columns` on a mapping.
- Keep table schema and column order identical to current output.

Acceptance:
- Output table equals current output for DFG/CDG inputs.

Status: Completed (constant columns now added via compute helpers).

## Workstream D: Dedupe Fallback Without `to_pylist`
Target: `src/codeintel/build/tabular/arrow_ops.py`

### D1. Compute-based dedupe fallback
Checklist:
- When `drop_duplicates` fails, append a row index column (`pa.array(range(n))`).
- Group by key columns and aggregate min row index (`group_by(...).aggregate([("row_idx","min")])`).
- Use an `is_in` mask over row indices and `table.filter(mask)` to keep first rows.
- Keep temporary row index internal (not surfaced to output).
- Keep Python fallback only if compute path fails.

Acceptance:
- Dedupe behavior matches existing Python fallback.
- No `to_pylist()` needed in the fallback path.

Status: Completed (compute fallback preserves order and keeps Python fallback).

## Workstream E: Tests and Validation

### E1. Storage query tests
Checklist:
- Add or update tests covering:
  - `safe_count_non_positive` numeric vs non-numeric columns.
  - `safe_count_duplicates` for string + numeric columns.
  - `safe_count_orphan_refs` with allow-null true/false.

Status: Covered by existing ingestion tests in `tests/ingestion/test_db_queries.py`; no new tests added.

### E2. Graph prefiltering tests
Checklist:
- Add minimal fixtures to assert outputs unchanged after filtering.
- Ensure column presence checks still behave the same.

Status: Partial (compute mask/filter unit tests added; no new end-to-end graph fixtures).

### E3. Dedupe fallback test
Checklist:
- Add a small table fixture where `drop_duplicates` raises and fallback is used.
- Validate row order and key uniqueness.

Status: Completed (added Arrow ops dedupe test).

### E4. Quality gates
Checklist:
- `uv run ruff check --fix` (for touched files).
- `uv run pyright --warnings --pythonversion=3.13` (touched files).
- `uv run pyrefly check` (touched files).
- Targeted pytest selection for affected modules.

Status: Targeted ruff/pyright/pyrefly passed. Pytest and full guardrails/quality report blocked by Hamilton runtime type mismatch (RecordBatchReader vs LazyFrame) in test/guardrail setup.

## Rollout Plan (Sequenced)
1) Add/extend compute helpers and masks (Workstream A4, B1).
2) Storage query compute refactor (Workstream A1-A3).
3) Graph prefiltering rollouts (Workstream B2-B6).
4) PDG constant columns (Workstream C1).
5) Arrow ops dedupe fallback (Workstream D1).
6) Tests and quality gates (Workstream E1-E4).

## Risks & Mitigations
- Kernel coverage gaps: keep Python fallback in all compute paths.
- Type variance (strings/nested): gate compute paths by data type checks.
- Behavioral drift: validate against existing tests and sample fixtures.

## Acceptance Criteria
- No behavioral regressions in graph outputs or dataset validation.
- All touched files pass Ruff, Pyright, and Pyrefly with zero errors.
- Tests for storage queries and dedupe fallbacks are updated or added.

## Additional Opportunities (Implemented)
- `src/codeintel/build/hamilton/native/graphs/cdg.py`: added compute prefilters for required columns
  (`function_goid_h128`, `block_id`, `block_idx`, `edge_kind`) before `to_pylist()` loops.
- `src/codeintel/build/hamilton/native/graphs/goids.py`: applied compute prefilter on AST node
  rows to allowed node types (`Module`, `ClassDef`, `FunctionDef`, `AsyncFunctionDef`) and
  validated `path/name/lineno` before Python iteration (module nodes allow null name/lineno).
- `src/codeintel/build/hamilton/native/graphs/call_graph.py` and `import_graph.py`: compute masks
  for non-empty `rel_path/module/qualname` are satisfied via `filter_python_modules` plus
  `filter_python_goids` enforcing non-empty `qualname` when present.
- `src/codeintel/storage/queries/parquet.py`: switched `is_in` to `SetLookupOptions(value_set=...)`
  and added a dataset-scanner filter path for `safe_count_non_positive` to avoid full column reads.
