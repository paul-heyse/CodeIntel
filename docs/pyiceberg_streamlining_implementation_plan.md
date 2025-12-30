# PyIceberg Streamlining Implementation Plan

## Intent

Streamline the CodeIntel codebase around PyIceberg as the canonical metadata, schema, and snapshot layer while
maintaining Hamilton DAG outputs as the source-of-truth for schema derivation and data contracts. Consolidate
scan/write/validation paths, standardize streaming behavior, and remove legacy or compatibility paths that
duplicate functionality.

## Scope Summary

- Unify schema and contract handling through `codeintel.core.schemas.contracts`.
- Standardize Iceberg scan and streaming conversion paths across build and serving.
- Centralize transaction, snapshot property, and write settings logic.
- Consolidate tombstone filtering and enforcement across serving engines.
- Promote Iceberg metadata and statistics as the sole observability source.
- Use typed metadata cache for planning and guardrails.
- Decommission dataset manifest and legacy Arrow dataset paths after cutover.

## Phase 0: Inventory, Guardrails, and Cutover Flags

Goal: Create a complete inventory of compatibility/legacy surfaces and define cutover flags before code changes.

- Inventory and tag legacy pathways that bypass Iceberg:
  - `src/codeintel/storage/manifests/*`
  - `src/codeintel/storage/datasets/*`
  - `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
  - `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`
  - `src/codeintel/storage/serving/snapshot_service.py`
- Ensure guardrail flags are explicit and consistent:
  - `ICEBERG_READ_ENABLED`, `ICEBERG_WRITE_ENABLED`, `ICEBERG_TOMBSTONES_ENABLED`.
  - Decide default values for non-prod vs prod.
- Acceptance:
  - A checklist of legacy paths with owning modules and a removal timeline.
  - Guardrail settings documented in `codeintel.yaml` and env variable docs.

## Phase 1: Contract and Schema Unification

Goal: Make Arrow/JSON schema conversions and metadata composition single-surface.

- Centralize all schema conversions in `src/codeintel/core/schemas/contracts.py`.
- Ensure Iceberg field IDs and name mapping metadata are only injected via the contracts module.
- Remove any direct `pyarrow.Schema` construction in build/serving that bypasses contracts.
- Update:
  - `src/codeintel/core/iceberg/schema.py` to call contracts for metadata binding only.
  - `src/codeintel/build/hamilton/materializers/iceberg_saver.py` to use contract-derived metadata.
  - `src/codeintel/storage/iceberg/cache.py` to store Arrow IPC via `contracts.encode_schema_ipc`.
- Acceptance:
  - Only contracts module composes schema metadata.
  - Arrow IPC metadata roundtrips preserve Iceberg schema IDs and name mappings.

## Phase 2: Unified Scan + ColumnarStream Surface

Goal: One scan path for Iceberg reads with consistent streaming semantics.

- Make `src/codeintel/serving/semantic/iceberg_scans.py` the only scan resolver.
- Route all Iceberg reads (serving + build loaders) through `DataScan` and `ColumnarStream`:
  - `src/codeintel/build/hamilton/native/patterns/loaders.py` → `DataScan` only.
  - `src/codeintel/core/columnar/tabular_adapter.py` is the single conversion bridge.
- Remove ad-hoc `to_arrow()` or eager table materializations unless explicitly requested.
- Acceptance:
  - All Iceberg reads use `DataScan.to_arrow_batch_reader()` for streaming.
  - No divergent scan logic in serving engines.

## Phase 3: Write Path Consolidation

Goal: Single transaction and snapshot property flow for Iceberg writes.

- Centralize snapshot property creation in one function and reuse across:
  - `IcebergDatasetSaver`
  - Migration/backfill helpers
  - Tombstone writes
- Use `Table.transaction()` for co-committing schema/spec updates and writes.
- Consolidate write settings (compression, row-group, dictionary) and always persist them to snapshot properties.
- Ensure external ingest path uses `Table.add_files` and does not rewrite Parquet by default.
- Acceptance:
  - All writes include deterministic snapshot properties.
  - Schema/spec/sort updates are always committed atomically.

## Phase 4: Tombstone Enforcement Consolidation

Goal: Single tombstone filter implementation with consistent scoping rules.

- Make `src/codeintel/serving/semantic/tombstones.py` the only SQLGlot transform.
- Ensure both DuckDB and Polars engines call the same tombstone filter logic.
- Standardize tombstone scoping:
  - `snapshot_id <= serving_snapshot_id` when available.
  - Warn but continue when tombstone table is missing during rollout.
- Acceptance:
  - One canonical filter implementation.
  - Engines share consistent tombstone behavior.

## Phase 5: Observability + Statistics

Goal: Iceberg metadata and statistics are the authoritative source for validation and drift.

- Use `table.inspect` + `table.update_statistics()` as the sole stats source.
- Emit stats via `observation_codec` and remove Parquet stats reliance.
- Align drift/validation to Iceberg schema IDs and snapshot IDs.
- Acceptance:
  - Observation payloads include Iceberg snapshot/schema IDs.
  - Validation no longer depends on Parquet metadata.

## Phase 6: Metadata Cache as Planner Backbone

Goal: Typed metadata cache is the query planner input for serving and CLI.

- Refresh `metadata.iceberg_*` on:
  - Build completion
  - Serving snapshot load
  - Explicit CLI refresh command
- Standardize all metadata queries via SQLGlot AST generation.
- Acceptance:
  - No direct inspection queries bypass the metadata cache.
  - Cache refresh is observable and deterministic.

## Phase 7: Guardrails + Settings Consolidation

Goal: Single source of truth for Iceberg policy and IO configuration.

- Normalize `IcebergSettings` configuration for:
  - FileIO impl
  - IO options
  - Location provider behavior
- Enforce guardrails at build and serving boundaries with explicit errors.
- Acceptance:
  - Guardrail errors are consistent and actionable.
  - All IO configs originate from `SettingsView`.

## Phase 8: Decommission Legacy and Compatibility Code

Goal: Remove duplicated legacy paths after staged cutover.

Decommission targets (after Iceberg cutover validation):
- Dataset manifests:
  - `src/codeintel/storage/manifests/*`
  - `src/codeintel/storage/datasets/arrow_store.py`
  - `src/codeintel/storage/serving/snapshot_service.py`
- Arrow dataset build paths:
  - `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
  - `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`
- Manifest migration helpers:
  - `src/codeintel/storage/datasets/maintenance.py` (replace with Iceberg migration only)

Compatibility code to retire:
- Fallback logic that chooses Arrow dataset path when Iceberg is enabled.
- Any direct Parquet stats capture outside Iceberg metadata.
- Any schema inference path that ignores Iceberg field IDs or name mapping.

Acceptance:
- All legacy modules removed or isolated behind a single deprecated adapter.
- Documentation updated with new single-path architecture.

## Phase 9: Tests and Cutover (Fast Cutover, No Dual-Write)

Goal: Enforce Iceberg-only behavior with comprehensive tests.

- Add integration tests:
  - `write → snapshot → read` with tombstones.
  - Schema evolution and name mapping persistence.
  - Time travel reads via `serving/<env>` refs.
- Remove or rewrite tests that assume manifest/Arrow dataset behavior.
- Cutover plan (fast, no dual-write):
  - Implement and validate all Iceberg read/write/serve paths in design-time tests.
  - Remove legacy manifest and Arrow dataset paths before first production use.
  - Enable Iceberg read/write flags once all cutover tests pass.

Acceptance:
- Tests cover all Iceberg read/write/serve paths.
- Legacy modules removed without gaps in coverage.

### Fast Cutover Checklist (Exact Tests + Removals)

Tests to add or update (must pass before cutover):
- `tests/build/hamilton/test_iceberg_materializer.py`
  - End-to-end: Hamilton → Iceberg write → snapshot properties → read via DataScan.
  - Schema evolution: field ID stability + name mapping persistence.
  - Tombstone diff: full snapshot rebuild emits tombstones and anti-join filters.
- `tests/serving/semantic/test_iceberg_scans.py`
  - Iceberg scan planning with SQLGlot filters + pushdown coverage.
  - Snapshot ref resolution (`serving/<env>`, `run/<run_id>`, `commit/<sha>`, fallback `main`).
- `tests/serving/semantic/test_tombstone_filtering.py`
  - SQLGlot anti-join injection is idempotent and snapshot-scoped.
  - Missing tombstone table warns but does not fail (when enabled).
- `tests/storage/test_iceberg_cache.py`
  - Metadata cache refresh on build completion and serving snapshot load.
  - Arrow IPC schema cache entries match contract metadata.
- `tests/storage/test_iceberg_stats.py`
  - Stats payload includes snapshot/schema IDs + tombstone metrics.
  - Statistics file persistence path returns a valid `StatisticsFile`.
- `tests/cli/test_iceberg_cli.py`
  - `iceberg.inspect`, `iceberg.refs`, `iceberg.manage-snapshots`, `iceberg.expire-snapshots`.

Legacy removals (delete or replace with Iceberg-only path):
- `src/codeintel/storage/manifests/*` (dataset manifest model + IO + registry).
- `src/codeintel/storage/datasets/arrow_store.py` (Arrow dataset stats + manifest writer).
- `src/codeintel/storage/serving/snapshot_service.py` (manifest-based serving snapshot prep).
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`
- `src/codeintel/storage/datasets/maintenance.py` (manifest maintenance utilities).
- Any remaining fallback branches that select Arrow dataset paths when Iceberg is enabled.

Test updates (remove legacy assumptions):
- Remove tests that assert dataset manifest payloads or Parquet stats metadata.
- Replace manifest-based snapshot tests with Iceberg snapshot + ref checks.
- Ensure all table materialization tests validate `iceberg_snapshot_id` in results.

Cutover gates:
- All above tests passing locally with `ICEBERG_READ_ENABLED=true` and `ICEBERG_WRITE_ENABLED=true`.
- No code references to dataset manifest modules remain (`rg -n "dataset_manifest|arrow_store|manifests"`).
- Guardrails ensure all enforced tables require Iceberg read/write.

## Success Criteria

- Zero dataset manifest reads for migrated tables.
- All serving reads are Iceberg `DataScan` streams.
- Schema drift reports reference Iceberg schema IDs.
- Codebase reduced by removal of legacy dataset manifest stack.
- Hamilton-derived schema and Iceberg metadata remain aligned by contract.
