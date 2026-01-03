# Boundary Audit: Build/Core/Config/CLI Dataflow

## Scope
- Focus: Arrow, Polars, Pandera, msgspec boundaries across build, core, config, and CLI.
- Goal: find non-value-add boundaries, duplicate validation/schema handling, and
  fan-out/fan-in data shuffles.

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

### 2) Forced materialization in `tabular_to_lazyframe`
- `src/codeintel/build/tabular/conversion.py` converts any `Iterable` input to a
  list before creating a `RecordBatchReader`, which forces full materialization.
- Why it is a boundary: upstream code can pass generators or streaming batches,
  but the conversion collapses them into a list.
- Candidate action:
  - Accept `Iterable[RecordBatch]` and build a streaming reader without
    materializing the whole list.

### 3) Duplicate dataset scanning implementations
- `src/codeintel/core/datasets/scanning.py` and
  `src/codeintel/storage/datasets/scanning.py` are effectively identical.
- Why it is a boundary: two copies create drift and unclear ownership.
- Candidate action:
  - Consolidate into a single canonical module (core) and re-export where
    needed.

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

### 5) Schema authority split between generated rows and Hamilton output lists
- Canonical column order is defined in generated row models:
  `src/codeintel/core/schemas/generated_rows/graph.py`
- CPG outputs also define column lists in Hamilton:
  `src/codeintel/build/hamilton/native/graphs/cpg.py`
- Why it is a boundary: schema and output selection are duplicated.
- Candidate action:
  - Derive `_CPG_NODE_COLUMNS` and `_CPG_EDGE_COLUMNS` from the generated row
    models (or SchemaService) to avoid drift.

### 6) Parallel schema alignment in build helpers and core
- Build alignment: `src/codeintel/build/tabular/frames.py`
- Core alignment: `src/codeintel/core/columnar/schema_alignment.py`
- Why it is a boundary: build re-implements contract alignment logic that core
  already owns.
- Candidate action:
  - Keep alignment logic in core; build should only delegate.

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

## Recommended next actions (high value)
1) Collapse duplicate scanning modules by consolidating into core.
2) Replace eager `Iterable -> list` conversion in `tabular_to_lazyframe`.
3) For Python-loop-heavy nodes, pick a single format (Arrow or Polars) and keep
   it through the node, avoiding back-and-forth conversions.
4) Make generated row models the single source of truth for column selection in
   Hamilton outputs (no duplicate column lists).
5) Define a single validation boundary and make other layers consume results.

## Notes
- This audit identifies boundaries and suggests consolidation targets. It does
  not propose code changes yet.
