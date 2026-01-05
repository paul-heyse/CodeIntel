# Build Metadata Decoupling Plan (Build-First, Storage Read-Only)

## Goals
- Make `build` fully independent of `storage` and DuckDB metadata tables.
- Move contract catalog and run metadata generation into build outputs under `build/`.
- Treat `storage` as a read-only consumer of build outputs (Parquet + build metadata bundle).
- Replace static configuration with programmatic derivation wherever possible.

## Non-Goals
- Redesign dataset schemas or rename table keys.
- Remove declared schemas entirely (retain as fallback for non-inferable or external tables).
- Change build semantics beyond decoupling and metadata placement.

## Current State Summary
- Build emits a metadata bundle under `build/metadata/` (contract catalog, schema manifest/registry,
  schema versions/observations, dataflow/lineage, run reports/index, export audit).
- Storage supports read-only bundle ingest (`codeintel meta sync` and
  `codeintel storage ingest-metadata`).
- Export audit, schema versions, observations, dataflow, and lineage are written to the bundle;
  contract catalog and schema manifest are hashed into canonical catalogs during ingest.
- Build CLI still opens a storage gateway by default (`codeintel build run` requires gateway).
- Core build paths still depend on gateway reads for schema inference settings, asset tracking, and
  fallback dataset counts (e.g., `arrow_dataset_saver`, asset emitter).
- Legacy gateway writes remain when a metadata bundle is not configured (build run metadata still
  persists to storage in that path).
- Schema observation resolution still uses gateway-only providers (bundle-only observation provider
  wiring is not complete).
- Contract resolution still depends on static configuration more than intended.

## Target State Summary
- Build emits a **Build Metadata Bundle** under `build/metadata/` that becomes the source of truth.
- Storage loads build outputs and metadata bundle; no build-time persistence into DuckDB.
- Contract catalog, schema registry, schema observations, and run metadata are derived programmatically.
- Static overrides remain optional and minimal (only when inference cannot produce required metadata).

## Plan Conventions
- No module owners or hand-offs; the plan is end-to-end for a single implementation effort.
- Rollback planning is intentionally omitted per request.

## Build Metadata Bundle (New Build Artifact Set)

### Base Layout (all under `build/`)
- `build/metadata/bundle_manifest.json`
- `build/metadata/contracts/contract_catalog.json`
- `build/metadata/contracts/contract_catalog.hash`
- `build/metadata/schema/schema_manifest.json`
- `build/metadata/schema/schema_registry.json`
- `build/metadata/schema/schema_versions.jsonl`
- `build/metadata/schema/schema_observations.jsonl`
- `build/metadata/lineage/derived_edges.jsonl`
- `build/metadata/lineage/derived_columns.jsonl`
- `build/metadata/dataflow/dataset_nodes.jsonl`
- `build/metadata/dataflow/dataset_edges.jsonl`
- `build/metadata/runs/run_report_<run_id>.jsonl`
- `build/metadata/runs/run_index.jsonl`
- `build/metadata/exports/export_audit.jsonl`

### Canonical Payloads
- **Contract catalog**: `version`, `contracts` mapping keyed by table_key.
- **Schema registry**: current schema pointer per table_key plus inference status.
- **Schema versions**: content-addressed schema records, including JSON schema and renderer cache.
- **Schema observations**: Arrow IPC schema + stats derived from dataset manifests.
- **Run reports**: build run status + output catalog (already emitted by build).
- **Lineage/dataflow**: derived from DAG edges + manifest-level dependency info.

### Bundle Manifest
- `bundle_manifest.json` is the root descriptor for the bundle.
- Fields:
  - `bundle_schema_version` (string, e.g. "v1")
  - `generated_at` (ISO8601 UTC string)
  - `repo`, `commit`, `run_id`
  - `files` (list of objects: `path`, `sha256`, `size_bytes`, `record_count`, `schema_version`)

### Artifact Schemas (v1)

#### `contracts/contract_catalog.json`
- `version` (int)
- `generated_at` (ISO8601)
- `repo`, `commit`
- `contracts` (object mapping `table_key` -> payload)
- Payloads use `contract_payload_to_json_obj(...)` output to preserve the existing
  DatasetContract JSON shape.

#### `contracts/contract_catalog.hash`
- `sha256` digest of `contract_catalog.json` (single-line text).

#### `schema/schema_manifest.json`
- Use the existing `compile_schema_manifest` JSON output (unchanged).

#### `schema/schema_registry.json`
- `version` (int)
- `generated_at` (ISO8601)
- `repo`, `commit`
- `entries` (list of `TableSchemaRegistryRecord` JSON objects):
  - `table_key`, `schema_digest`, `schema_hash`
  - `derivation_kind`, `derivation_source`
  - `inference_status`, `inference_error`
  - `catalog_hash`, `updated_at`

#### `schema/schema_versions.jsonl`
- JSONL of `SchemaVersionRecord`:
  - `schema_digest`, `schema_hash`, `schema_json`
  - `renderer_cache`, `created_at`

#### `schema/schema_observations.jsonl`
- JSONL of `SchemaObservationRecord`:
  - `observation_id`, `table_key`, `repo`, `commit`, `target_name`
  - `schema_digest`, `schema_hash`, `arrow_schema_ipc_b64`
  - `column_stats`, `dataset_stats`, `derived_settings`, `drift_summary`
  - `observed_at`

#### `lineage/derived_edges.jsonl`
- `repo`, `commit`, `downstream`, `upstream`, `edge_type`
- `source` (e.g. "dag", "manifest")
- `created_at`

#### `lineage/derived_columns.jsonl`
- `repo`, `commit`, `downstream_table`, `downstream_column`
- `upstream_table`, `upstream_column`, `edge_type`
- `source`, `created_at`

#### `dataflow/dataset_nodes.jsonl`
- `id`, `kind`, `family`, `owner_package`, `description`

#### `dataflow/dataset_edges.jsonl`
- `src`, `dst`, `edge_type`

#### `runs/run_report_<run_id>.jsonl`
- Existing run report records:
  - `run_metadata`, `tag_schema_summary`, `output_catalog`
- Preserve current record shapes for compatibility.

#### `runs/run_index.jsonl`
- `run_id`, `repo`, `commit`, `started_at`, `duration_ms`, `success`
- `report_path`, `computed_targets_count`, `failed_targets_count`

#### `exports/export_audit.jsonl`
- `dataset`, `macro`, `rows`, `duration_s`, `output_path`, `sql`, `plan`, `created_at`

## Bundle Versioning and Compatibility
- Add `bundle_schema_version` to `bundle_manifest.json`.
- Bundle consumers must reject unsupported versions with a clear error message.
- Allow additive fields in JSON/JSONL payloads with forward-compat parsing.
- Hash `contract_catalog.json` and `schema_manifest.json` to detect drift.

## Programmatic Derivation Rules

### Dataset Contract Fields (Derived by Default)
- `table_key`: from Hamilton output tags or catalog.
- `name`: suffix of `table_key`.
- `schema`: inferred from Arrow schema in dataset manifests (fallback: declared schema).
- `jsonl_filename` / `parquet_filename`: derived from `table_key` naming convention.
- `family`: prefix of `table_key`.
- `owner_package`: derived from prefix or tags.
- `tags`: from Hamilton tags; only allow explicit overrides when present.
- `validation_profile`: derived by policy (domain-based default) or tag override.
- `schema_version`: derived from schema hash/digest.
- `upstream_dependencies`: derived from DAG edges or lineage artifacts.
- `is_view`: `docs.*` prefix or view tag.

### Schema Registry + Observations
- Use `SchemaIndex` + inference observations from build runtime.
- If a table does not exist in outputs, use declared schema only.
- Schema observations are derived from dataset manifests and Arrow schema metadata.

### Run Metadata
- Use build run report JSONL as canonical source; no DuckDB writes.
- Index `run_report_<run_id>.jsonl` in a `run_index.jsonl` for fast lookup.

### Derivation Rules Table (Contract Fields)
| Field | Primary Source | Derivation Logic | Fallback |
| --- | --- | --- | --- |
| table_key | DAG output tags | Use `table_key` tag | Error if missing |
| name | table_key | suffix of `schema.table` | None |
| schema | Arrow manifest | parse Arrow schema -> TableSchema | declared schema |
| jsonl_filename | policy | `${table_key}.jsonl` | tag override |
| parquet_filename | policy | `${table_key}.parquet` | tag override |
| family | table_key | prefix of `schema.table` | "core" |
| owner_package | table_key | prefix or tag | None |
| tags | DAG tags | include `ci.*` tags | empty set |
| validation_profile | policy | by domain | tag override |
| schema_version | schema | hash/digest -> version | None |
| upstream_dependencies | DAG edges | downstream deps | empty list |
| is_view | table_key | `docs.*` or tag | False |

## Storage Ingest Contract (Read-Only)
Status: implemented for schema/contracts/lineage/dataflow/export audit; run report ingest pending.
- New module: `src/codeintel/storage/metadata/ingest.py`.
- API surface:
  - `load_build_metadata_bundle(bundle_root: Path, con: DuckDBPyConnection) -> IngestReport`
  - `validate_build_metadata_bundle(bundle_root: Path) -> BundleValidation`
  - `bundle_manifest_from_path(bundle_root: Path) -> BundleManifest`
- Ingest flow:
  - Validate `bundle_manifest.json` and file hashes.
  - Stream JSONL into DuckDB tables using Arrow readers.
  - Upsert by primary keys; replace-by-run for run reports.
  - Preserve idempotency for repeated ingests.

## CLI and Runtime Wiring
Status: partial (meta sync + storage ingest implemented; build run decoupling pending).
- `codeintel build run`:
  - Default: no storage gateway; emits metadata bundle.
  - Optional: `--emit-metadata-bundle` (default true).
- `codeintel meta sync`:
  - New behavior: ingest build metadata bundle.
  - Optional: `--bundle-root` to point to a non-default location.
- `codeintel storage ingest-metadata`:
  - Explicit ingest command for operators.

## Cutover Strategy (Sharp)
- [x] Build emits metadata bundle by default under `build/metadata/`.
- [x] Storage ingest consumes the bundle via `codeintel meta sync` or
  `codeintel storage ingest-metadata`.
- [ ] Remove legacy gateway writes and fallback readers (Phase 6).
- [ ] Remove legacy metadata locations and update docs once cutover is complete.

## Failure Modes and Recovery
- Missing bundle: emit clear error and instruct to re-run build.
- Corrupt bundle: reject ingest and report offending file/hash.
- Partial run: ingest run report with `success=false`, skip schema registry updates.
- Schema mismatch: mark `inference_error` and keep previous registry entries.

## Performance and Size Budget
- JSONL emission uses streaming writes (no in-memory accumulation).
- Ingest uses Arrow readers with batch size tunables.
- Bundle size goal: < 200 MB for full repo (excluding Parquet datasets).

## Observability
- Log `build.metadata.bundle_written` with path, run_id, size, file count.
- Log `storage.metadata.bundle_ingest` with duration, row counts, errors.
- Include `bundle_schema_version` and `catalog_hash` in logs.

## Edge Cases
- Targets producing zero rows (schema observation from empty manifest).
- Views without SQL plans (skip view SQL diff in schema manifest inputs).
- Missing dataset manifests (record inference error; keep contract entry).
- Multiple runs for same commit (run index stores all run_ids).

## Validation Policy Update
- Manifest-based validation in build phase.
- Storage validation reads schema registry from bundle.
- If schema observation missing, fall back to declared schema with warning.

## Implementation Plan

Status legend: [x] complete, [ ] pending, [-] intentionally omitted.

### Phase 0: Audit (omitted)
- [-] Skipped per request.

### Phase 1: Build Metadata Bundle Writers
- [x] Add build module `src/codeintel/build/meta/bundle.py` to emit metadata bundle.
- [x] Implement `bundle_manifest.json` hashing and file list assembly.
- [x] Emit `contract_catalog.json` directly from build contract service.
- [x] Emit schema manifest + registry without calling storage gateway.
- [x] Emit schema versions and schema observations JSONL.
- [x] Emit lineage/dataflow artifacts (dataset nodes/edges, derived edges/columns).
- [x] Emit run index from run reports.

### Phase 2: Decouple `serving_artifacts` from Storage
- [x] Replace gateway persistence in `serving_artifacts` with build artifact files and bundle emission.
- [x] Keep JSON artifacts (`semantic_registry.json`, `schema_manifest.json`, `buildspec.json`).
- [x] Emit `schema_manifest.json` only to build artifacts, not storage.
- [x] Write catalog inputs/metadata as part of the build bundle instead of DuckDB.

### Phase 3: Storage Loader (Read-Only Ingest)
- [x] Add storage ingestion module `src/codeintel/storage/metadata/ingest.py`.
- [x] Implement bundle validation (hash checks).
- [x] Load build metadata bundle into DuckDB tables as a read-only import step.
- [x] Update `codeintel meta sync` to call the new ingest logic (no build-time dependencies).
- [x] Add `codeintel storage ingest-metadata` for explicit operator ingest.
- [ ] Ingest run reports and run index into build/run metadata tables (define tables + upsert rules).
- [ ] Enforce bundle schema version compatibility and required file presence.
- [ ] Update gateway open logic to permit a no-catalog mode for build-only workflows.

### Phase 4: Contract Provider Simplification
- [ ] Implement programmatic defaults in contract resolution (derive family/name, filenames, owner).
- [ ] Derive schema + schema_version from manifests/observations with declared schema fallback.
- [ ] Wire bundle-only schema observation provider for schema resolution (no gateway dependency).
- [ ] Keep tag overrides only for exceptions; document supported override tags.
- [ ] Reduce reliance on static configuration in `config/datasets` to non-inferable cases.
- [ ] Document which tables remain declared-only and why.

### Phase 5: CLI + Runtime Wiring
- [ ] Update build CLI to avoid opening a storage gateway by default (gateway on-demand only).
- [ ] Split build commands that still require gateway (publish, assets, history) to open on demand.
- [ ] Ensure build execution uses metadata bundle and never calls gateway persistence paths.
- [x] Add a storage-only CLI path to ingest build artifacts into DuckDB.
- [ ] Ensure read-only storage access operates solely from build outputs + metadata bundle.
- [x] Add explicit error messaging when bundle is missing.

### Phase 6: Legacy Removal (Sharp Cutover)
- [ ] Remove legacy storage writes from build (run records, schema registry, catalog writes).
- [ ] Remove gateway-based schema observation resolution and asset tracking from build.
- [ ] Delete fallback logic that reads legacy metadata locations.
- [ ] Remove legacy config flags and unused tables/DDL tied only to write-time paths.
- [ ] Prune docs/plan references to legacy metadata locations.

## API and Data Model Changes
- New build metadata bundle format under `build/metadata/`.
- New storage ingestion API for build metadata.
- Contract derivation policy (domain-based defaults + tag overrides).

## Testing and Validation
- [ ] Unit tests for contract derivation from build outputs.
- [ ] Schema registry/manifest round-trip tests from bundle.
- [ ] Storage ingestion tests that rebuild DuckDB metadata from bundle alone.
- [ ] Run report + run index ingestion tests (including upsert semantics).
- [ ] End-to-end build + storage ingest without any build-time storage gateway.
- [ ] Negative tests for corrupt bundle, missing required files, and unsupported bundle versions.

## Acceptance Criteria
- Build can run without opening a storage gateway.
- Storage can fully reconstruct metadata tables by ingesting build outputs.
- Contract catalog is reproducible from build outputs and matches existing contracts.
- Run metadata and schema registry are derived from build outputs, not DuckDB state.
- No static configuration required unless inference is impossible.

## Open Questions
- Which non-inferable tables require declared schemas permanently?
- How to encode lineage edges for custom targets that do not expose dataset manifests?
- Versioning strategy for bundle schemas (semantic version or digest-based).
- Which build/run metadata tables should ingest run reports and run index, and what is the
  upsert/replace policy?
