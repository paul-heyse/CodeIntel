# Boundary Audit: Build/Core/Config/CLI Dataflow

## Scope
- Focus: Arrow, Polars, Pandera, msgspec boundaries across build, core, config, and CLI.
- Goal: find non-value-add boundaries, duplicate validation/schema handling, and
  fan-out/fan-in data shuffles.

## Status Summary (current)
- Completed
  - Core dataset scanning consolidation (storage re-exports core).
  - `tabular_to_lazyframe` streaming iterable support (no eager list materialization).
  - Schema alignment centralized in core (build delegates).
  - CPG column authority derived from generated row models.
- In progress
  - Reducing Arrow -> Polars -> Arrow materialization in Hamilton nodes.
  - Validation boundary consolidation to avoid double-run checks.

## Findings (prioritized)

### 1) Repeated Arrow -> Polars -> Arrow conversions and eager collects
- Pattern: `tabular_to_lazyframe(...).collect()` in many Hamilton nodes, followed by
  row-based Python logic and then re-emitting LazyFrames.
- Examples:
  - `src/codeintel/build/hamilton/native/graphs/goids.py`
  - `src/codeintel/build/hamilton/native/graphs/call_graph.py`
  - `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
  - `src/codeintel/build/hamilton/native/graphs/cdg.py`
  - `src/codeintel/build/hamilton/native/graphs/symbol_use.py`
  - `src/codeintel/build/hamilton/native/analytics/semantic_roles.py`
  - `src/codeintel/build/hamilton/native/analytics/data_models.py`
  - `src/codeintel/build/hamilton/native/analytics/entrypoints.py`
  - `src/codeintel/build/hamilton/native/analytics/function_effects.py`
- Why it is a boundary: data is materialized into Polars DataFrames for Python
  loops, then converted back to LazyFrame for downstream materialization.
- Candidate action:
  - If the computation can be expressed with Polars expressions, keep LazyFrame
    end-to-end and avoid `collect()`.
  - If Python loops are required, prefer Arrow-native iteration on record
    batches or `pa.Table.to_pylist()` and return Arrow readers instead of
    round-tripping through LazyFrame.
- Status: in progress.
  - Completed: config graph metrics, config data flow, subsystem agreement/cache
    now use Polars-native paths and avoid row dict assembly in Hamilton.
  - Remaining checklist:
    - [ ] `src/codeintel/build/hamilton/native/graphs/goids.py`: replace
      `iter_rows` with Polars joins/aggregations; keep LazyFrame until final
      `rows_to_frame`.
    - [ ] `src/codeintel/build/hamilton/native/graphs/call_graph.py`: build edge
      weights and node attrs via Polars group_by; avoid per-row Python updates.
    - [ ] `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`: vectorize AST
      node and GOID lookups (Polars joins or Arrow batch map); avoid
      `iter_rows`.
    - [ ] `src/codeintel/build/hamilton/native/graphs/cdg.py`: move block/edge
      assembly to Polars expressions and grouped aggregation.
    - [ ] `src/codeintel/build/hamilton/native/graphs/symbol_use.py`: replace
      module/goid/occurrence loops with Polars join + explode pipelines.
    - [ ] `src/codeintel/build/hamilton/native/analytics/semantic_roles.py`:
      convert module + features iteration to Polars joins; avoid row dicts.
    - [ ] `src/codeintel/build/hamilton/native/analytics/data_models.py`:
      vectorize module resolution and row assembly with Polars expressions.
    - [ ] `src/codeintel/build/hamilton/native/analytics/entrypoints.py`:
      replace module/feature iteration with Polars join + aggregation.
    - [ ] `src/codeintel/build/hamilton/native/analytics/function_effects.py`:
      minimize `tabular_to_frame` materialization; keep Arrow/Polars streaming
      where possible and limit Python loops to AST extraction only.

### 2) Forced materialization in `tabular_to_lazyframe`
- `src/codeintel/build/tabular/conversion.py` converts any `Iterable` input to a
  list before creating a `RecordBatchReader`, which forces full materialization.
- Why it is a boundary: upstream code can pass generators or streaming batches,
  but the conversion collapses them into a list.
- Candidate action:
  - Accept `Iterable[RecordBatch]` and build a streaming reader without
    materializing the whole list.
- Status: completed.
  - Implemented streaming iterable conversion via `RecordBatchReader` without
    list materialization.

### 3) Duplicate dataset scanning implementations
- `src/codeintel/core/datasets/scanning.py` and
  `src/codeintel/storage/datasets/scanning.py` are effectively identical.
- Why it is a boundary: two copies create drift and unclear ownership.
- Candidate action:
  - Consolidate into a single canonical module (core) and re-export where
    needed.
- Status: completed.
  - Storage now re-exports the core scanning helpers.

### 4) Validation entry points that can double-run checks
- Build validation: `src/codeintel/build/hamilton/data_quality.py`
- Storage validation: `src/codeintel/storage/validation/columnar.py`
- Analytics utility validation: `src/codeintel/build/analytics/utilities/datasets.py`
- Export validation: `src/codeintel/build/exports/validation.py`
- Why it is a boundary: if build and storage validation are both invoked for the
  same dataset, Pandera and constraint checks can run twice.
- Candidate action:
  - Establish a single canonical validation boundary (build or storage) and
    have the other path consume the validation result rather than re-run it.
- Status: in progress.
  - Completed: analytics dataset validation now delegates to
    `storage.validation.columnar.validate_table`.
  - Remaining checklist:
    - [ ] Decide canonical boundary (storage is current candidate) and update
      build/exports/hamilton paths to consume validation results instead of
      re-running.
    - [ ] `src/codeintel/build/hamilton/data_quality.py`: route to storage
      validation context or emit metadata marking datasets as validated.
    - [ ] `src/codeintel/build/exports/validation.py`: reuse storage validation
      context for Parquet/JSONL checks where possible; avoid double Pandera runs.
    - [ ] `src/codeintel/build/hamilton/native/patterns/savers.py` and
      `src/codeintel/build/exports/engine.py`: ensure `validation_profile`
      metadata is plumbed once and not validated twice downstream.

### 5) Schema authority split between generated rows and Hamilton output lists
- Canonical column order is defined in generated row models:
  `src/codeintel/core/schemas/generated_rows/graph.py`
- CPG outputs also define column lists in Hamilton:
  `src/codeintel/build/hamilton/native/graphs/cpg.py`
- Why it is a boundary: schema and output selection are duplicated.
- Candidate action:
  - Derive `_CPG_NODE_COLUMNS` and `_CPG_EDGE_COLUMNS` from the generated row
    models (or SchemaService) to avoid drift.
- Status: completed.
  - CPG column selection now derives from `columns_for_table_key`.

### 6) Parallel schema alignment in build helpers and core
- Build alignment: `src/codeintel/build/tabular/frames.py`
- Core alignment: `src/codeintel/core/columnar/schema_alignment.py`
- Why it is a boundary: build re-implements contract alignment logic that core
  already owns.
- Candidate action:
  - Keep alignment logic in core; build should only delegate.
- Status: completed.
  - Build frame alignment now delegates to core alignment helpers.

## Fan-out / fan-in shuffles to target
- Graph and analytics pipelines frequently:
  1) Convert inputs to LazyFrame.
  2) `collect()` to DataFrame.
  3) Perform Python loops.
  4) Re-emit LazyFrame for materialization.
- Representative examples:
  - `src/codeintel/build/hamilton/native/graphs/goids.py`
  - `src/codeintel/build/hamilton/native/graphs/call_graph.py`
  - `src/codeintel/build/hamilton/native/analytics/data_models.py`
- Status: still active in the listed nodes; see remaining checklist in item 1.

## Recommended next actions (high value)
1) Convert remaining Hamilton nodes in item 1 to Polars/Arrow-native pipelines.
2) Finalize validation boundary decision and wire build/exports/hamilton paths
   to avoid duplicate checks.

## Notes
- This audit identifies boundaries and suggests consolidation targets. It does
  not propose code changes yet.
