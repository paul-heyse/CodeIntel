# Build Join Optimization Plan (Polars, PyArrow, SQLGlot)

## Goal
Improve join performance and determinism across `src/codeintel/build` by:
- Enforcing join correctness (cardinality validation, schema alignment).
- Reducing intermediate materialization costs (pushdown + streaming scans).
- Centralizing join logic via SQLGlot where SQL is already the orchestration layer.

## Non-goals
- No changes to storage or serving layers beyond build-time join behavior.
- No new database engines.
- No new schema sources (Hamilton remains authoritative).

## Primary Hotspots (Join-heavy paths)
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/build/hamilton/native/graphs/cpg.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/views/view_outputs.py`
- `src/codeintel/build/graphs/engine/views.py`

## Phase 0: Baseline and Guardrails
**Purpose:** Ensure correctness before speeding things up.

Checklist:
- Identify joins that must be 1:1 or m:1 (unique keys) and annotate them.
- Confirm any span-join ordering assumptions (bytes vs lines) and document in code.
- Add simple counters for join match rates (not tests) to log once per run.

Files and actions:
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
  - Document join key contracts for `_symbol_goid_xref_frame` and `_occurrence_span_xref_frame`.
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
  - Document join key contracts for `_occurrence_resolution_frame` and `_resolve_facts`.
- `src/codeintel/build/hamilton/native/graphs/cpg.py`
  - Document join key contracts for `_occurrence_roles` and fallback joins.

Acceptance:
- Each of the above files clearly labels join key constraints and expected cardinality.

## Phase 1: Polars Join Validation + Multi-collect
**Purpose:** Use Polars to validate correctness and reduce redundant execution.

Checklist:
- Add `validate=` to joins with expected cardinality.
- Replace sequential collects on related LazyFrames with `pl.collect_all(...)`.
- Use `collect_schema()` only when needed; avoid forcing full `collect()` during planning.

Files and actions:
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
  - Add `validate="m:1"` or `"1:1"` where symbol/key uniqueness is expected.
  - Keep `collect()` only for spans used in Python-side indexing.
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
  - Use `pl.collect_all([facts_with_bytes, facts_without_bytes, occ_bytes])` to avoid repeated scans.
  - Add `validate` on joins where byte spans are expected to be unique.
- `src/codeintel/build/hamilton/native/graphs/cpg.py`
  - In `_occurrence_roles`, collect `span` and `syntax` via `pl.collect_all`.
  - Add `validate` on `syntax.join(span, ...)`.
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
  - Add `validate` on arg-to-param joins where cardinality is known.
  - Consider `partition_by(["repo", "commit", "rel_path"])` before heavy join sequences.

Acceptance:
- All expected-unique joins explicitly validate cardinality.
- Join pipelines that used multiple `.collect()` now use `pl.collect_all()`.

## Phase 2: PyArrow Join/Scan Optimizations
**Purpose:** Avoid loading unnecessary data and speed up join-heavy flows.

Checklist:
- Use `pyarrow.dataset.Scanner` for projection/filter pushdown.
- Convert to Arrow `Table` for heavy joins before Polars when it reduces overhead.
- Use `Table.join` with `coalesce_keys=True` and `filter_expression` for residual filters.

Files and actions:
- `src/codeintel/build/graphs/engine/views.py`
  - When scanning snapshots, use Arrow dataset scans with projection on just join keys.
  - Convert to Polars only after scan + minimal join columns are loaded.
- `src/codeintel/build/hamilton/native/graphs/cpg.py`
  - For `occ_syntax` + `occ_span` joins, consider Arrow join path before Polars if it reduces materialization.
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
  - If joins are dominated by large Parquet scans, add an Arrow scan path to filter by `repo/commit` first.

Acceptance:
- Any dataset scan feeding a join uses Arrow projection/filter pushdown where possible.

## Phase 3: SQLGlot-Driven Join Construction
**Purpose:** Normalize join logic and reduce ad-hoc SQL string handling.

Checklist:
- Construct join-heavy views via SQLGlot AST, not raw SQL.
- Use SQLGlot optimizer on view ASTs before execution.
- Use SQLGlot metadata extraction to audit join keys and referenced tables.

Files and actions:
- `src/codeintel/build/hamilton/native/views/view_outputs.py`
  - For view definitions returned as SQL strings, parse to AST, then rewrite and re-render.
  - Optionally optimize via `sqlglot.optimizer.optimize` with schema hints.
- `src/codeintel/core/sqlglot_tools.py`
  - Add helper to extract join keys and validate expected join shape.

Acceptance:
- All build views render SQL through SQLGlot AST; join metadata is extractable.

## Cross-cutting Enhancements
- Add a small helper in build utils for “join shape validation” (reuse across modules).
- Add a single place to document join policies (1:1 vs m:1) for core tables.

## Rollout Strategy
1) Phase 0 + Phase 1 in one PR (low risk, mostly validation and execution planning).
2) Phase 2 Arrow scan pushdowns in targeted hotspots.
3) Phase 3 SQLGlot AST adoption for view joins.

## Validation
- Run the Hamilton targets that exercise SCIP, syntax enrich, and CPG joins.
- Verify join validation does not raise on known good data.
- Confirm materialized row counts are stable (within expected deltas).

## Risks and Mitigations
- **Join validation failures**: may expose prior hidden data inconsistencies.
  - Mitigation: log offenders, gate with feature flags if needed.
- **Arrow scan differences**: might change default null handling or type casting.
  - Mitigation: compare schema inference and use explicit casts.
- **SQLGlot optimization changes**: may alter view SQL semantics.
  - Mitigation: enable optimizer only where safe and compare output SQL.
