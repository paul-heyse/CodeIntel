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

## Status Update (Completed Scope)
- Default dataset root now resolves to `build/datasets` under the repo root (via
  `BuildPaths.dataset_root_dir`) and is configurable via runtime/CLI overrides.
- StorageConfig now carries parquet-mode options and dataset root context; the
  config is threaded through gateway creation, CLI gateway openers, pool wiring,
  and serving DB hot-swap management.
- Dataset manifests are attached to `DatasetRegistry` when gateways open.
- Parquet-backed relation resolution is the default for dataset tables, with
  parquet-only enforcement and manifest-based file list scanning.
- DuckDB contract schema resolution falls back to dataset manifests when tables
  are missing.
- Export validation checks for dataset manifests instead of DuckDB tables.
- DuckDB dataset writes are blocked under parquet-only policy.
- Build-side analytics writes now land in parquet datasets with manifest output.
- Build/serving table existence checks now consult dataset manifests.
- New CLI command `datasets migrate-parquet` added to materialize parquet
  snapshots from legacy DuckDB tables.
- Migration workflow supports optional legacy table drops after export.
- Serving DB manager warns on missing dataset manifests and fails fast when
  manifest roots cannot be resolved.
- Documentation updates landed in `docs/architecture.md` and
  `docs/dataset_root_configuration.md`.
- Unit and integration tests cover parquet relation resolution, migration,
  exports, warehouse reads, and serving snapshot preparation.

## Implementation Plan

### Phase 0: Path and Configuration Hardening
- [x] Define the canonical dataset root via `BuildPaths.dataset_root_dir` and
  provide runtime/CLI overrides.
- [x] Extend `StorageConfig` with dataset root context and thread through gateway
  creation, CLI gateway openers, pool wiring, and serving DB manager.
- [x] Attach dataset manifests to the dataset registry when gateways open.

### Phase 1: Parquet-Backed Relation Resolution
- [x] Resolve dataset relations from manifests using the parquet scan adapter,
  including manifest file lists and partitioning metadata.
- [x] Prefer parquet relations for dataset tables, with fallback to
  `con.table` only when dataset context is absent.
- [x] Enforce parquet-only policy to fail fast when manifests are missing.
- [x] Update build preflight/executor table checks to use manifests.

### Phase 2: Build Output and Export Alignment
- [x] Block legacy DuckDB dataset writes under parquet-only policy (policy backend
  enforcement + analytics utility guard).
- [x] Replace `information_schema` checks with dataset manifest validation in
  the export pipeline.
- [x] Export pipeline already uses gateway relations; parquet backing is now
  inherited from the resolver without semantic changes.
- [x] Audit remaining build-side write paths for direct DuckDB inserts and
  migrate them to `write_dataset` with manifest generation.
- [x] Confirm all build dataset outputs are manifest-backed (no silent skips).

### Phase 3: Storage and Serving Read Path Updates
- [x] DuckDB schema resolution falls back to dataset manifest schema.
- [x] Warehouse/schema helpers carry dataset root context for parquet lookup.
- [x] Serving DB manager passes dataset root context into pool storage config.
- [x] Verify serving snapshot view registration is manifest-aware and points to
  parquet scans when datasets are present.
- [x] Add explicit failure messaging when serving opens without manifests under
  parquet-only policy (serve-time diagnostics).

### Phase 4: Enforcement, Cleanup, and Migration
- [x] Prevent dataset writes to DuckDB under parquet-only policy.
- [x] Add `datasets migrate-parquet` CLI command for one-time DuckDB export.
- [x] Document parquet-only data path and dataset root configuration.
- [x] Add CLI docs for migration usage, safeguards, and expected output
  structure (manifest + parquet layout).
- [x] Add optional cleanup workflow (archive/drop legacy DuckDB dataset tables)
  after successful migration verification.

## Testing and Validation
- [x] Unit: parquet-backed relation resolution (paths, columns, partitioning).
- [x] Unit: manifest loading + error handling under parquet-only policy.
- [x] Unit: migration command writes manifest metadata + file lists correctly.
- [x] Integration: build export workflow reads from parquet, not DuckDB tables.
- [x] Integration: warehouse read APIs function without dataset tables in DuckDB.
- [x] Integration: serving snapshot creation succeeds with only parquet datasets.

## Acceptance Criteria
- No build dataset is stored in DuckDB tables.
- DuckDB reads build datasets exclusively via parquet scans.
- Export pipeline and warehouse reads operate correctly without dataset tables.
- Dataset manifests are required and validated for all build datasets.

## Risks and Mitigations
- Missing manifests: fail fast under parquet-only policy; add migration tooling.
- Path transition risk: allow explicit dataset root override during rollout.
- Performance regressions: tune `from_parquet` options and leverage
  partitioning metadata from manifests.
