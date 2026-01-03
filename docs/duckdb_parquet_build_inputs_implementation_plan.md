# DuckDB Parquet Build Inputs Implementation Plan

## Purpose
Commit to a parquet-first integration between build outputs and DuckDB. Any data
that originates from `src/codeintel/build` and is later queried in DuckDB must be
persisted as parquet under the dataset store and then scanned by DuckDB. DuckDB
no longer owns dataset table storage; it only scans parquet and holds metadata.

## Decisions (Target State)
- Build outputs are written only as parquet datasets with manifests.
- DuckDB reads build datasets via `parquet_scan` / `from_parquet` using dataset
  manifests; DuckDB tables are not a required persistence layer for build data.
- Metadata, audit logs, and registry tables may remain in DuckDB.
- The dataset store lives under the storage datasets subsystem and is the
  canonical path for all build dataset outputs.

## Non-Goals
- Removing DuckDB entirely.
- Rewriting serving query planners.
- Changing the public dataset contract model (table keys, schemas, manifests).

## Inventory: DuckDB Inputs Originating from Build
The following paths currently read build-produced datasets through DuckDB tables:

1. Relation access for dataset tables
   - `src/codeintel/storage/gateway/relation.py`
   - `src/codeintel/storage/gateway/accessors.py`
   - `src/codeintel/storage/warehouse.py`
   - `src/codeintel/storage/repositories/*`

2. Build export pipeline (JSONL / Parquet)
   - `src/codeintel/build/exports/exprs.py`
   - `src/codeintel/build/exports/common.py`
   - `src/codeintel/build/exports/engine.py`
   - `src/codeintel/build/exports/jsonl.py`
   - `src/codeintel/build/exports/parquet.py`

3. Schema and registry checks that assume DuckDB tables exist
   - `src/codeintel/storage/schema/duckdb_contracts.py`
   - `src/codeintel/build/exports/common.py` (information_schema checks)

4. Legacy DuckDB insert helpers (build-side writes)
   - `src/codeintel/build/analytics/utilities/datasets.py`

## Target Architecture
- Dataset outputs are persisted via the storage datasets API:
  - `codeintel.storage.datasets.arrow_store.write_dataset`
  - `codeintel.storage.datasets.manifests.write_dataset_manifest`
- Dataset manifests are loaded for the active snapshot and attached to the
  dataset registry.
- DuckDB relations for dataset tables are built from parquet scans using the
  manifest file list, partitioning info, and schema metadata.
- DuckDB retains only metadata tables (registry, audit, run tracking).

## Implementation Plan

### Phase 0: Path and Configuration Hardening
1. Define the canonical dataset root:
   - Update `src/codeintel/config/primitives.py` to set
     `BuildPaths.dataset_root_dir` to the storage datasets root.
   - Add a path override in config / CLI if needed so legacy paths can be
     explicitly retained during migration.
2. Extend storage configuration to carry dataset context:
   - Add `dataset_root_dir` and `snapshot_id` (commit) to
     `src/codeintel/storage/gateway/config.py:StorageConfig`.
   - Thread these through gateway open paths in:
     `src/codeintel/storage/gateway/factory.py`,
     `src/codeintel/cli/services/storage.py`,
     `src/codeintel/cli/handlers/_utilities.py`,
     `src/codeintel/serving/db/manager.py`.
3. Load dataset manifests into the registry on gateway creation:
   - Use `codeintel.storage.datasets.manifest_index.load_dataset_manifests`
     plus `attach_dataset_manifests` to populate
     `DatasetRegistry.dataset_manifests`.

### Phase 1: Parquet-Backed Relation Resolution
1. Create a dataset relation resolver:
   - New helper in `src/codeintel/storage/datasets` (or
     `src/codeintel/storage/gateway/relation.py`) to produce
     `DuckDBRelation` from dataset manifests using
     `src/codeintel/serving/semantic/duckdb_scan_adapter.py`.
   - Use manifest file lists and partitioning to configure `from_parquet`.
2. Update `DuckDBGateway.relation_from_table_key` to prefer parquet:
   - If the table key exists in `datasets.by_table_key` and a manifest is
     available, return a parquet scan relation.
   - If manifest is missing but dataset root exists, resolve a manifest
     from the dataset snapshot directory.
   - If no dataset context is present, fall back to `con.table` (metadata
     and non-build tables).
3. Add a hard enforcement mode:
   - Introduce a config flag (e.g., `dataset_source="parquet_only"`) to
     forbid `con.table` for dataset table keys and raise if manifests are
     missing. This ensures a hard commit to parquet-backed inputs.

### Phase 2: Build Output and Export Alignment
1. Ensure all build outputs write parquet + manifest:
   - Standardize on `codeintel.storage.datasets.arrow_store.write_dataset`.
   - Remove or deprecate direct DuckDB insertion helpers in
     `src/codeintel/build/analytics/utilities/datasets.py`.
2. Update export validation to use dataset manifests:
   - Replace `information_schema` checks in
     `src/codeintel/build/exports/common.py` with manifest existence checks.
3. Export pipeline uses parquet-backed relations:
   - Update `src/codeintel/build/exports/exprs.py` to rely on the new
     relation resolver (not raw DuckDB tables).
   - Validate that `build_export_relation_plan` is unchanged in semantics
     and only switches input source.

### Phase 3: Storage and Serving Read Path Updates
1. Schema resolution from parquet:
   - In `src/codeintel/storage/schema/duckdb_contracts.py`, fall back to
     dataset manifest schema when DuckDB tables are absent.
2. Warehouse and repository queries:
   - Ensure `src/codeintel/storage/warehouse.py` and repository helpers use
     parquet-backed relations through the gateway.
3. Serving snapshot preparation:
   - Reuse dataset view registration logic from
     `src/codeintel/storage/serving/snapshot_service.py` to build views that
     point to parquet scans when needed.

### Phase 4: Enforcement, Cleanup, and Migration
1. Prevent dataset writes to DuckDB:
   - Add checks in `DuckDBPolicyBackend` to block `ensure_table` /
     `bulk_insert` for dataset tables when parquet-only mode is enabled.
2. Migration path for existing DuckDB tables:
   - Add a one-time export command that materializes parquet datasets from
     DuckDB tables and writes manifests under the dataset root.
   - Optionally drop or ignore the DuckDB tables after verification.
3. Update docs:
   - Document the parquet-only data path in `docs/architecture.md` and add
     a short operational guide for dataset root configuration.

## Testing and Validation
- Unit tests:
  - Parquet-backed relation resolution (scan paths, columns, partitioning).
  - Manifest loading and error handling in parquet-only mode.
- Integration tests:
  - Build export workflow reads from parquet, not DuckDB tables.
  - Warehouse read APIs function without dataset tables in DuckDB.
- Regression checks:
  - Serving snapshot creation succeeds when only parquet datasets exist.

## Acceptance Criteria
- No build dataset is stored in DuckDB tables.
- DuckDB reads build datasets exclusively via parquet scans.
- Export pipeline and warehouse reads operate correctly without dataset tables.
- Dataset manifests are required and validated for all build datasets.

## Risks and Mitigations
- Missing manifests: fail fast in parquet-only mode; add migration tooling.
- Path transition risk: allow explicit dataset root override during rollout.
- Performance regressions: tune `from_parquet` options and leverage
  partitioning metadata from manifests.
