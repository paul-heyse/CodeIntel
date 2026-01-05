# Arrow-First Build Plan for CPG Joins + Scans

## Goal
Move CPG and related graph assembly toward Arrow-first joins/scans, leaving
Polars only for final materialization or operations that Arrow does not cover.

## Scope
- Arrow-first join utilities (build/tabular).
- Arrow-first scan utilities (dataset scanning).
- One end-to-end CPG subgraph conversion as a template.

## Phase 1: Arrow-First Join Utilities
**Files**
- `src/codeintel/build/tabular/arrow_ops.py`
  - Arrow join helpers (`arrow_join_tables`, `arrow_join_frames`).
  - Arrow table materialization helpers (`arrow_table_from_tabular`, `arrow_table_from_lazyframe`).

**Follow-ups**
- Add optional `validate` parameter with explicit uniqueness checks for:
  - `m:1`: right keys unique.
  - `1:1`: both sides unique.
  - `1:m`: left keys unique.
- Add `suffix` support for overlapping non-key columns.

## Phase 2: Arrow-First Scan Utilities
**Files**
- `src/codeintel/build/graphs/engine/datasets.py`
  - Add `scan_snapshot_table(...) -> pa.Table | None` that wraps `scan_snapshot_reader`.
  - Add `scan_snapshot_reader_with_columns(...)` for repeated call sites.
- `src/codeintel/build/tabular/conversion.py`
  - Add `tabular_to_arrow_table(...)` for caller convenience and type reuse.

**Targets**
- Use `scan_snapshot_table` in graph view loaders for call/import/metrics graphs.
- Keep projection/filter pushdown via Arrow dataset scanner options.

## Phase 3: CPG Template Conversion (Arrow-First)
**Template target**
- `src/codeintel/build/hamilton/native/graphs/cpg.py`

**Conversion**
- `_occurrence_roles`:
  - Use `pl.collect_all([span_lf, syntax_lf])`.
  - Join via Arrow (`arrow_join_frames`) instead of Polars join.
  - Preserve fallback span matching logic and payload assembly.

**Validation**
- Keep join contract comment in `_occurrence_roles`.
- If we add `validate` to Arrow joins, enforce `m:1` for span rows.

## Phase 4: Rollout to Additional CPG Subgraphs
**Candidate functions**
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
  - Arg/param joins and binding enrichment.
- `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
  - Symbol resolution joins.
- `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
  - Block/edge joins on bytecode facts.
- `src/codeintel/build/hamilton/native/graphs/pdg.py`
  - Dataflow wiring joins.

**Pattern**
1. Convert input LazyFrames to Arrow tables via `arrow_table_from_lazyframe`.
2. Join via `arrow_join_tables` with explicit `keys` and `how`.
3. Convert back to Polars only at output boundaries.

## Phase 5: Ingestion Join Conversions (Arrow-First)
**Files**
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`

**Pattern**
- Scan or collect minimal columns into Arrow tables.
- Apply Arrow joins, then only materialize Polars for post-join transforms.

## Validation Checklist
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- Verify join output row counts are stable in CPG and graph outputs.
- Confirm schema compatibility after Arrow joins (no unexpected nulls or type drift).

