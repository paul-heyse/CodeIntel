# Arrow-First Build Maximization Plan

## Goal
Make `pyarrow` the canonical tabular type across `src/codeintel/build`, so
graph construction runs on Arrow tables/readers end-to-end and only materializes
Polars data at the final view/export boundary (if at all).

## Non-goals
- No changes to storage or serving layers beyond build-time data flow.
- No schema source changes (Hamilton remains the single authority).
- No behavioral changes to graph semantics beyond improved performance and stability.

## Target state (definition of done)
- All graph nodes in `src/codeintel/build/hamilton/native/graphs/*` operate on
  `pyarrow.Table` or `pyarrow.RecordBatchReader`.
- Join/scan/aggregate logic is centralized in Arrow utilities (no ad-hoc
  Polars joins for graph construction).
- Views and exports only convert to Polars when explicitly needed for UI or
  formatting, otherwise remain Arrow-native.

---

## Status update (completed)
- Arrow utilities extended (`align_table_to_contract`, parquet scan helpers, Arrow dedupe).
- Graph modules converted to Arrow outputs: `import_graph.py`, `call_graph.py`,
  `goids.py`, `cfg_dfg.py`, `pdg.py`, `symbol_use.py`.
- `cpg.py` + `call_wiring.py` now emit Arrow readers at the boundary (internal
  joins still include Polars).
- `call_wiring.py` now performs call target joins in Arrow tables (remaining
  internal joins still include Polars).
- `cpg.py` + `call_wiring.py` now run Arrow-first end-to-end (joins and transforms).
- Ingestion targets now Arrow-first:
  - `ingest_targets.py` emits Arrow readers.
  - `extraction_targets.py`, `syntax_augment.py`, `syntax_enrich.py` emit Arrow readers.
  - Ingestion pipeline cleanup/normalize now Arrow-based.
- `_coerce_none_output` now returns `empty_reader_for_table` (Arrow-only fallback).
- Views + exports now Arrow-native:
  - `view_outputs.py` loads/produces `RecordBatchReader` using DuckDB over Arrow tables.
  - `graphs/engine/views.py` scans Arrow readers directly (no Polars dependency).
  - `exports/engine.py`, `exports/common.py`, `exports/jsonl.py`, `exports/parquet.py`
    read Parquet snapshots as Arrow readers and write Arrow-native outputs.
- `cdg.py` now emits Arrow readers end-to-end.
- Join coverage logs added to `call_graph.py` and `goids.py`.
- Arrow join policy documented (`docs/arrow_join_policy.md`).
- Arrow-first guard script + CI workflow added (`scripts/ci/arrow_first_guard.sh`,
  `.github/workflows/arrow-first-guard.yml`).

---

## Phase 0: Inventory + Guardrails
**Purpose:** Establish scope, contracts, and invariant checks before conversion.

Checklist:
- Inventory all join-like operations in `src/codeintel/build` and classify
  each by join keys + expected cardinality.
- Define the Arrow-first data contract for build graphs:
  - Input: `pa.Table` or `pa.RecordBatchReader`.
  - Output: `pa.Table` or `pa.RecordBatchReader`.
  - Alignment: always aligned to Hamilton contract schema before join.
- Add a single "join policy" reference doc (keys + cardinality) in the plan
  or in a shared module docstring.

File targets:
- `src/codeintel/build/hamilton/native/graphs/*.py`
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/tabular/arrow_ops.py`

Acceptance:
- Join policy inventory exists with explicit keys and expected shapes.
- A single Arrow-first contract is documented and referenced by build graphs.

Status: done.

---

## Phase 1: Arrow Utilities Core (Foundation)
**Purpose:** Make Arrow joins/scans a first-class, shared capability.

Checklist:
- Implement or extend `src/codeintel/build/tabular/arrow_ops.py` to include:
  - `scan_parquet_dataset(...)` with projection + predicate pushdown.
  - `align_to_contract(...)` to enforce schema ordering and extras handling.
  - `join_tables(...)` with inner/left/semi/anti support and key validation.
  - `group_by_aggregate(...)` for common graph reductions.
  - `coalesce_columns(...)` for multi-source joins.
- Ensure Arrow ops support both `pa.Table` and `pa.RecordBatchReader`.
- Update `src/codeintel/build/tabular/frames.py` to convert Arrow -> Polars
  only at the explicit view/export boundary.

File targets:
- `src/codeintel/build/tabular/arrow_ops.py`
- `src/codeintel/build/tabular/frames.py`
- `src/codeintel/build/tabular/conversion.py`
- `src/codeintel/core/columnar/schema_alignment.py`

Acceptance:
- All graph conversions use Arrow utilities instead of local join logic.
- Arrow utilities handle schema alignment and extras policy consistently.

---

## Phase 2: Import Graph (Template Conversion)
**Purpose:** Convert a low-complexity graph to Arrow-first as the template.

Checklist:
- Replace any `pl.LazyFrame` usage in `src/codeintel/build/hamilton/native/graphs/import_graph.py`
  with `pa.Table` or `pa.RecordBatchReader`.
- Use `arrow_ops.join_tables` for all joins in import graph.
- Ensure contract alignment for each Arrow table before joins.
- Document join key expectations in import graph (inline docstring comments).

File targets:
- `src/codeintel/build/hamilton/native/graphs/import_graph.py`

Acceptance:
- Import graph executes without Polars in the compute path.
- Join outputs match existing row counts and schema columns.

Status: done.

---

## Phase 3: Call Graph + GOIDs (High-Value Joins)
**Purpose:** Move high fan-out joins to Arrow-first for major downstream impact.

Checklist:
- Convert `src/codeintel/build/hamilton/native/graphs/call_graph.py` to Arrow-only:
  - Replace join sequences with `arrow_ops.join_tables`.
  - Replace per-join column selection with Arrow projection.
- Convert `src/codeintel/build/hamilton/native/graphs/goids.py` to Arrow-only:
  - Apply Arrow joins for symbol/GOID mapping.
  - Ensure hash/key columns are aligned to contract schemas.
- Add metrics for join coverage (match rates) as non-failing logs.

File targets:
- `src/codeintel/build/hamilton/native/graphs/call_graph.py`
- `src/codeintel/build/hamilton/native/graphs/goids.py`

Acceptance:
- Call graph + GOID joins run without Polars in compute steps.
- Join coverage metrics show stable match rates.

Status: done (join coverage metrics still optional).

---

## Phase 4: CPG Graph (Join-Heavy Conversion)
**Purpose:** Move the core CPG graph pipeline to Arrow-first.

Checklist:
- Replace join-heavy sequences in `src/codeintel/build/hamilton/native/graphs/cpg.py`
  with Arrow joins (occurrences, syntax nodes, roles, call edges).
- Where joins were previously Polars + Python-side fallbacks, implement
  Arrow-first equivalents or isolate a single controlled fallback boundary.
- Align all Arrow tables to contract schemas before each join.

File targets:
- `src/codeintel/build/hamilton/native/graphs/cpg.py`
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
- `src/codeintel/build/hamilton/native/graphs/pdg.py`
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`

Acceptance:
- CPG graph functions are Arrow-only end-to-end.
- Join outputs are stable vs prior baseline (row counts and key coverage).

Status: done.

---

## Phase 5: Views + Exports (Arrow-Native Boundary)
**Purpose:** Keep Arrow as the transport to views/exports and minimize
unnecessary Polars conversion.

Checklist:
- Update `src/codeintel/build/graphs/engine/views.py` to use Arrow scans
  and Arrow joins; only convert to Polars for final presentation.
- Update `src/codeintel/build/hamilton/native/views/view_outputs.py` to
  accept Arrow inputs and convert at the latest possible point.
- Ensure exporters (`src/codeintel/build/exports/*`) accept Arrow tables
  directly and avoid intermediate Polars collections where not needed.

File targets:
- `src/codeintel/build/graphs/engine/views.py`
- `src/codeintel/build/hamilton/native/views/view_outputs.py`
- `src/codeintel/build/exports/engine.py`
- `src/codeintel/build/exports/parquet.py`
- `src/codeintel/build/exports/jsonl.py`
- `src/codeintel/build/exports/common.py`

Acceptance:
- Views and exports accept Arrow tables/readers without intermediate Polars
  collections, except where required for UI formatting.

Status: done.

---

## Phase 6: Enforcement + Cleanup
**Purpose:** Prevent regression to Polars in compute paths.

Checklist:
- Add lintable checks (or simple `rg` audit in CI) to flag `pl.LazyFrame`
  usage in graph compute modules.
- Document the Arrow-first policy in `src/codeintel/build/tabular/arrow_ops.py`
  and `src/codeintel/build/tabular/frames.py`.
- Remove unused Polars imports in build graph modules.

Acceptance:
- No Polars usage remains in graph compute code.
- Arrow-first policy is documented and enforced.

Status: done.

---

## Validation Plan (per phase)
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  after each phase.
- Execute the impacted Hamilton targets per phase:
  - Phase 2: `import_graph`
  - Phase 3: `call_graph`, `goids`
  - Phase 4: `cpg`, `cfg_dfg`, `pdg`
  - Phase 5: `view_outputs`, `exports`
- Compare row counts and key coverage against baseline.

---

## Risks and Mitigations
- **Arrow join semantics differ from Polars**:
  - Mitigation: explicitly align schema + types; add join coverage metrics.
- **Performance regressions on small datasets**:
  - Mitigation: add size-based routing if needed (Arrow for large, Polars for tiny).
- **Schema drift or missing columns**:
  - Mitigation: always align to Hamilton contract schema before joins.

---

## Rollout Strategy
1) Phase 0 + Phase 1 (foundation).
2) Phase 2 (import_graph as template).
3) Phase 3 (call_graph + goids).
4) Phase 4 (cpg + remaining graphs).
5) Phase 5 + Phase 6 (views/exports + enforcement).
