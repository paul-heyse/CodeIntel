# Hamilton Full-Cache Materialization Plan (Arrow-First, Non-Streaming Outputs)

## Goals
- Make every intermediate Hamilton node cacheable with deterministic hashes.
- Remove `pyarrow.RecordBatchReader` outputs from the build DAG in `src/codeintel/build`.
- Standardize on fully materialized Arrow tables (`pa.Table`) as the tabular contract.
- Preserve Arrow-first data handling while prioritizing cache correctness over streaming.

## Non-Goals
- Rewriting dataset schemas or table keys.
- Changing graph semantics or target selection behavior.
- Optimizing for lowest RAM usage (we accept higher memory to gain caching).
- Introducing dynamic execution or changing executor topology.

## Current State (Summary)
- Many nodes return `pa.RecordBatchReader`, which is single-pass and unhashable.
- Hamilton caching warns about unhashable outputs (non-deterministic data versions).
- Cache adapter already materializes Arrow readers for result storage but returns readers,
  so hashing still sees unhashable stream types.
- Some nodes are explicitly `@cache(behavior="disable")` even though they are pure transforms.

## Target State (Definition)
- All intermediate nodes return `pa.Table` (or other hashable, materialized types).
- Hamilton data versioning has a deterministic hashing path for `pa.Table`.
- Caching is enabled for all intermediate nodes (non-materialization, non-side-effect).
- Materialization nodes (export, dataset saves) remain `RECOMPUTE`.
- No unhashable warnings during `codeintel build run --all`.

## Design Decisions
- **Tabular contract:** `pa.Table` is the canonical intermediate representation.
- **Hashing strategy:** deterministic hash over materialized table content
  (schema + row data) using Arrow IPC serialization or batch-wise hashing.
- **Caching scope:** all intermediate compute nodes are cacheable; I/O or side-effect
  nodes remain `RECOMPUTE`, and pure context nodes use `IGNORE` to avoid key pollution.

## Implementation Plan

### Phase 1: Add Deterministic Hashing for Arrow Tables
- [ ] Add `codeintel.build.hamilton.arrow_hashing` (or similar) module that registers
  `hamilton.caching.fingerprinting.hash_value` for:
  - `pa.Table`
  - `pa.RecordBatch` (if still encountered during transition)
  - `pa.Schema` (optional, for ancillary hashing)
- [ ] Choose the hashing algorithm:
  - Option A (simple, deterministic): serialize table via Arrow IPC stream and hash bytes.
  - Option B (streaming digest): iterate record batches and update a hasher to reduce peak RAM.
- [ ] Add unit tests to validate deterministic hashes across runs.

### Phase 2: Normalize Cache Adapter to Return Tables
- [ ] Update `ManifestBackedCacheAdapter.do_node_execute` to return `pa.Table`
  for Arrow results (not `RecordBatchReader`) when caching behavior allows.
- [ ] Ensure `ArrowFileResultStore` and `ArrowCachedResult` continue to store/load
  `pa.Table` and can rehydrate to tables without streaming.
- [ ] Add a safety guard: if a node returns `pa.RecordBatchReader` after the migration,
  log a warning and materialize before hashing.

### Phase 3: Convert Tabular Helpers to Table-First APIs
- [ ] Update `codeintel.build.tabular.conversion` to make `tabular_to_arrow_table`
  the default for intermediate operations.
- [ ] Introduce explicit helpers for the few remaining streaming boundaries:
  - `ensure_table(...)` for materialized inputs.
  - `ensure_reader(...)` only at final I/O boundaries (exports/materializers).
- [ ] Update docstrings and typing annotations to reflect `pa.Table` as the default.

### Phase 4: Migrate DAG Nodes from Reader to Table
Apply systematically across these groups (non-exhaustive, but prioritized):

- [ ] **Ingestion targets**
  - `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
  - `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
  - `src/codeintel/build/hamilton/native/ingestion/scip*.py`
  - `src/codeintel/build/hamilton/native/ingestion/syntax_augment.py`
  - `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
  - `src/codeintel/build/hamilton/native/ingestion/file_line_index.py`

- [ ] **Graphs**
  - `src/codeintel/build/hamilton/native/graphs/*` (call_graph, import_graph, cfg_dfg, pdg, cdg, symbol_use)
  - `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
  - `src/codeintel/build/hamilton/native/graphs/cpg/_legacy.py` (retain correctness during migration)

- [ ] **Analytics**
  - `src/codeintel/build/hamilton/native/analytics/*` (graph_metrics, cfg_dfg_metrics, config_graphs,
    data_models, entrypoints, function_* modules, subsystem_* modules, validation)

- [ ] **Views**
  - `src/codeintel/build/hamilton/native/views/view_outputs.py` to return `pa.Table`
    (move `RecordBatchReader` to final materialization if needed).

For each module:
- [ ] Replace `pa.RecordBatchReader` return types with `pa.Table`.
- [ ] Replace `table_to_reader(...)` usage with direct `pa.Table` returns.
- [ ] Replace `tabular_to_arrow_reader(...)` inputs with `tabular_to_arrow_table(...)`.
- [ ] Update any `input_type=pa.RecordBatchReader` in table specs to `pa.Table`.

### Phase 5: Update Target Templates and Materializers
- [ ] Update `TableTargetTableSpec` usage to default to `pa.Table` (or `InferableTabularInput`
  that resolves to tables).
- [ ] Ensure `save_dataset` / `save_relation_table` accept `pa.Table` inputs directly.
- [ ] Review `arrow_dataset_saver.py` to remove streaming reader assumptions.

### Phase 6: Cache Behavior Policy Alignment
- [ ] Remove `@cache(behavior="disable")` from intermediate compute nodes that are deterministic.
- [ ] Keep `RECOMPUTE` on:
  - materialization/output nodes,
  - export targets,
  - any side-effect-only nodes.
- [ ] Apply `CachingBehavior.IGNORE` to non-data context nodes (e.g., `catalog`, `cache_index`,
  `cache_key_resolver`) to avoid cache key pollution.
- [ ] Ensure default cache behavior remains `default` for all compute nodes.

### Phase 7: Validation and Performance Checks
- [ ] Run `uv run codeintel build run --all --verbose=1` and confirm:
  - no unhashable warnings,
  - cache hit/miss logs are deterministic.
- [ ] Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- [ ] Validate dataset manifests and schema observations remain consistent.
- [ ] Compare cache sizes and runtime between two consecutive runs to confirm
  reuse of cached intermediates.

## Risks and Mitigations
- **Higher RAM usage:** mitigate with batch-wise hashing and chunked table assembly.
- **Longer initial run:** acceptable trade-off for cache hits on subsequent runs.
- **Hashing cost for large tables:** provide optional config to hash on dataset manifest
  (schema hash + row count + file hash) if needed later.

## Acceptance Criteria
- Full build completes with no unhashable warnings.
- All intermediate nodes produce deterministic cache keys and data versions.
- Second run yields high cache hit rate with identical outputs.
- No functional regressions in exported datasets or metadata bundle outputs.
