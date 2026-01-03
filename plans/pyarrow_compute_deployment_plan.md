# PyArrow Compute Deployment Plan (Build Pipeline)

## Overview
This plan introduces targeted PyArrow compute usage across `src/codeintel/build` to replace
Python row loops and `to_pylist()`/`from_pylist()` round-trips where we can safely stay
columnar. The focus is on vectorized filters, projections, and aggregations using
`pyarrow.compute` and dataset/table expressions as described in:

- `docs/python_library_reference/pyrarrow.md`
- `docs/python_library_reference/pyarrow-advanced.md`

The goal is to shrink Python work in graph-building pipelines, reduce per-row overhead, and
align with Arrow’s compute patterns while preserving current behavior and type safety.

## Goals
- Replace Python-level masks and row filters with Arrow compute expressions.
- Reduce `to_pylist()` usage to only cases that require Python AST analysis or bespoke logic.
- Standardize compute usage via `pc.call_function` to keep type checkers happy.
- Preserve schema alignment and current outputs (no behavioral regressions).

## Non-Goals
- Rewriting AST or graph construction logic into pure Arrow compute.
- Large-scale structural refactors of CPG/graph pipelines.
- Introducing new dependencies or changing dataset schemas.

## Compute Primitives We Will Use
- Function registry calls: `pc.call_function`, `pc.get_function`, `pc.list_functions`.
- Scalar expressions: `pc.field`, `pc.scalar`, `ds.field`, boolean algebra on expressions.
- Table filtering: `table.filter(expr)`, Dataset scanner filters.
- Aggregations: `pc.call_function("min"/"max"/"count"/"sum", ...)` with `ScalarAggregateOptions`.
- Sorting: `pc.SortOptions` + `pc.call_function("sort_indices", ...)`.
- Casting: `pc.CastOptions` when safe/explicit coercions are needed.

## Global Quality Gates
- Ruff: `uv run ruff check <touched files>`
- Pyright: `uv run pyright <touched files>`
- Pyrefly: `uv run pyrefly check <touched files>`
- Update or add tests for any modified behavior (existing tests in `tests/graphs`,
  `tests/ingestion`, and `tests/analytics`).

## Workstream A: Core Arrow Ops Helpers
Target: `src/codeintel/build/tabular/arrow_ops.py`

### A1. Replace Python uniqueness check with compute aggregate
Current: `_ensure_unique_keys` uses `group_by(...).aggregate(...).to_pylist()` and Python loop.

Checklist:
- Add a compute path to read `max(count)` without `to_pylist()`.
- Use `pc.call_function("max", [counts], options=ScalarAggregateOptions(skip_nulls=True))`.
- Preserve current error message behavior.
- Keep fallback for unsupported kernels.

Acceptance:
- No `to_pylist()` in `_ensure_unique_keys`.
- Behavior unchanged for tables with duplicate keys.

### A2. Introduce compute utility helpers (if repeated patterns emerge)
Checklist:
- Add local helper for `safe_scalar(name, args, options)` in `arrow_ops` or a small shared module.
- Use helper for min/max/count patterns to avoid duplicated try/excepts.
- Ensure helper returns Python scalars via `.as_py()` or `None`.

Acceptance:
- Helper is used in at least two call sites.
- Ruff/pyright/pyrefly pass.

## Workstream B: Graph Pipelines (Call/Import/Goid/CFG/DFG/Symbol Use)
Target modules:
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`
- `src/codeintel/build/hamilton/native/graphs/import_graph.py`
- `src/codeintel/build/hamilton/native/graphs/goids.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`

### B1. `call_graph.py` pre-filter modules/goids with compute
Checklist:
- Build an Arrow expression that filters `core.modules` to Python rows with valid
  `path`/`module`.
- Apply `table.filter(expr)` before `to_pylist()`.
- In `_call_graph_index_rows`, filter `core.goids` to `kind in ("function","method")`,
  `rel_path` not null, `goid_h128` not null, and language in {None, "python"}.
- Keep AST parsing in Python; only reduce the input rows.

Acceptance:
- `to_pylist()` remains, but operates on a reduced table.
- Output tables unchanged for existing test cases.

### B2. `import_graph.py` module filtering with compute
Checklist:
- Filter `core.modules` to valid Python modules with non-null `path` and `module`.
- Apply `table.filter(expr)` before iterating in Python.

Acceptance:
- Fewer rows processed in Python, no behavior change.

### B3. `goids.py` module frame filtering via compute
Checklist:
- Use compute expression to filter `path`, `module`, `language` to non-null and non-empty.
- Apply filter before `to_pylist()`.

Acceptance:
- `_module_frame` uses `table.filter(expr)`; no behavior change.

### B4. `cfg_dfg.py` filtering and mask creation via compute
Checklist:
- Replace `_filter_goids_table` Python mask with Arrow compute expression:
  `kind in ("function","method")` and `language is null or == "python"`.
- Use `pc.call_function("is_in", ...)`, `pc.call_function("equal", ...)`,
  `pc.call_function("is_null", ...)`, and `pc.call_function("or_kleene"/"and_kleene")`.
- Apply `table.filter(expr)` instead of building Python `mask`.

Acceptance:
- `_filter_goids_table` does not construct Python masks.

### B5. `symbol_use.py` pre-filter occurrences/goids with compute
Checklist:
- Use compute to filter rows where required columns are valid (symbol, rel_path, start_line).
- Apply `table.filter(expr)` before `to_pylist()`.

Acceptance:
- Output matches current logic, fewer rows in Python loops.

## Workstream C: PDG and Call Wiring
Target modules:
- `src/codeintel/build/hamilton/native/graphs/pdg.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`

### C1. `pdg.py` replace row augmentation with compute projections
Checklist:
- Replace `_dfg_edges_rows`/`_cdg_edges_rows` `to_pylist()` with columnar additions:
  `table.append_column("edge_kind", pc.scalar("DFG"))`, etc.
- Use `pa.nulls(table.num_rows)` or `pc.scalar(None)` for null columns.
- Concatenate tables with `pa.concat_tables`.
- Keep dedupe path unchanged.

Acceptance:
- No `to_pylist()` in `pdg.py`.
- Output schema identical to previous path.

### C2. `call_wiring.py` reduce `to_pylist()` hotspots
Checklist:
- Identify sections that only add constants or drop columns.
- Replace with columnar operations (`append_column`, `drop_columns`, `rename_columns`).
- Keep `to_pylist()` only where per-row Python logic is required.

Acceptance:
- At least one high-volume `to_pylist()` usage is removed.

## Workstream D: CPG (`cpg.py`) hot paths
Target: `src/codeintel/build/hamilton/native/graphs/cpg.py`

This file is large; scope changes must be surgical.

### D1. Replace constant-column row loops
Checklist:
- Identify table builds that only add fixed constants.
- Replace with compute projections or `Table.append_column`.
- Preserve ordering and contract alignment.

### D2. Pre-filter large tables before Python loops
Checklist:
- Add `table.filter(expr)` for `kind`/`language`/`rel_path` validations before loops.
- Ensure filters match existing conditional logic exactly.

Acceptance:
- Verified against existing graph test suite (targeted tests only).

## Workstream E: Shared Utilities (optional but recommended)
Target: `src/codeintel/build/tabular/frames.py` or a new helper in `build/tabular`.

Checklist:
- Introduce a small compute helper for:
  - `safe_filter(table, expr)` with exception handling.
  - `scalar_from_compute(name, args, options)` returning Python scalar.
- Use the helper in at least two modules for consistent handling.

Acceptance:
- Helper functions covered by small unit tests (if applicable).

## Testing Strategy
- Run targeted tests for each modified module.
  - Graphs: `tests/graphs` and `tests/ingestion` (subset tied to touched modules).
  - CPG-specific: `tests/graphs/test_engine_nx.py` and other CPG-related tests.
- Use `pytest -q` on relevant subsets, not full suite unless needed.

## Rollout Strategy
- Implement workstreams sequentially, starting with A → B → C → D.
- After each workstream, run the quality gates and a small targeted test subset.
- Keep changes per workstream small to reduce regression risk.

## Detailed Checklist Summary
- [ ] A1: Arrow uniqueness check via compute
- [ ] A2: Optional compute helpers
- [ ] B1: call_graph compute pre-filters
- [ ] B2: import_graph compute pre-filters
- [ ] B3: goids compute pre-filters
- [ ] B4: cfg_dfg compute mask/filter
- [ ] B5: symbol_use compute pre-filters
- [ ] C1: pdg compute projections (remove to_pylist)
- [ ] C2: call_wiring compute projections on constant-only transforms
- [ ] D1: cpg constant-column refactors
- [ ] D2: cpg pre-filtering before loops
- [ ] E1: shared compute utilities
- [ ] Quality gates (ruff/pyright/pyrefly) after each workstream

## Implementation Notes
- Prefer `pc.call_function` over direct `pc.sum`/`pc.min` to avoid typing issues.
- Use `pc.ScalarAggregateOptions(skip_nulls=...)` for min/max/count.
- Use `pc.is_null`, `pc.is_valid`, `pc.equal`, `pc.is_in`, and Kleene logic for
  null-safe boolean combinations.
- Keep `to_pylist()` only for AST parsing or when row-wise Python is unavoidable.

## Risks and Mitigations
- **Risk**: subtle changes in null-handling during compute filters.
  - Mitigation: mirror current conditional logic and use Kleene ops where appropriate.
- **Risk**: compute kernel availability for certain types.
  - Mitigation: preserve fallback paths or use table-level methods (`drop_duplicates`) first.
- **Risk**: schema drift in `from_pylist` replacements.
  - Mitigation: enforce schema alignment with contract helpers already in place.
