# Build Output DuckDB (View-only) — Minimal Schema + View Strategy

This document describes a **minimal, build-owned DuckDB database** that acts as a **view-only**
SQL interface over the build’s Parquet snapshot outputs.

The intent is to preserve the current **decoupled boundary** (Parquet + manifests as the canonical
interface) while enabling downstream components to integrate via **DuckDB ↔ DuckDB** attachment,
without adding storage-coupled write paths to `src/codeintel/build`.

## Goals

- Provide a **single-file DuckDB artifact** that is **tied to one build snapshot** (no multi-snapshot
  requirements yet).
- Keep the DB **view-only**: no persisted data duplication; only views + small metadata tables.
- Align metadata with the existing **dataset manifest** and **contract identity** system.
- Make downstream orchestration ergonomic: `ATTACH ... (READ_ONLY)` then query domain tables.
- Provide a **downstream-friendly catalog** that supports:
  - dataset discovery even when outputs are missing/partial
  - contract/manifest identity checks (hash/version/row counts)
  - “agent-facing” introspection without filesystem crawls

## Non-goals

- Multi-snapshot browsing, time travel, or retention policies.
- Replacing Parquet snapshots as the source of truth.
- Implementing serving-ready materialization in the build layer.

## Source-of-truth inputs (what the builder reads)

The build output DB should be generated from these existing artifacts:

1) **Arrow dataset manifests** (authoritative for snapshot layout)
- `dataset_manifest.json` per `(table_key, snapshot_id)` directory.
- Model: `ArrowDatasetManifest` in `src/codeintel/core/manifests.py`.
- Helpers: `src/codeintel/core/datasets/manifests.py`, `src/codeintel/core/datasets/paths.py`.

2) **Contract identity metadata** (authoritative for schema/version provenance)
- Stored in manifest `extras` by the Arrow dataset saver.
- See `_manifest_extras(...)` in `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`.

3) **BuildSpec (recommended, for “expected vs present” inventory)**
- BuildSpec already lists the datasets that are part of the intended build output surface.
- A build-output DB that only enumerates *present manifests* is insufficient for downstream systems
  that must handle **partial/failed runs**. BuildSpec provides the “expected set” so the DB can
  represent missing outputs explicitly.

4) **Dataset contracts / registry metadata** (optional, for rich naming/semantics)
- If you want stable `dataset_name → table_key` naming (beyond what BuildSpec provides), hydrate
  from contracts (`codeintel.build.schemas.iter_contracts`).

## File placement (recommended)

Because the DB’s views will reference snapshot Parquet paths, the build DB must be generated for
the filesystem environment it will be queried in (paths are embedded in view definitions).

Important constraint (DuckDB table functions): `parquet_scan(...)` does **not** support “lateral”
column/subquery parameters for file paths, so views cannot compute their scan path by looking up a
value in a metadata table at query time. The builder must emit **one view per dataset** with a
literal scan path.

Recommended default:

- `build/serving/artifacts/build_output.duckdb` (preferred for publish workflows)
  - This keeps the build-output DB co-located with other serving artifacts (`semantic_registry.json`,
    `schema_manifest.json`, `buildspec.json`) that are already used for publishing.

Alternative:

- `dataset_root_dir/build_output.duckdb` (convenient for local inspection near data)

If you need to run the DB in an environment where dataset paths differ, plan to **rebuild** the DB
for that environment (see Portability considerations).

## Minimal DuckDB schema (tables + views)

### Schema: `build_meta`

This schema contains **small, persisted tables** that make the build DB self-describing.

#### Table: `build_meta.snapshot`

One row describing the snapshot this DB targets.

Suggested columns (minimal):

- `run_id TEXT NULL` (build run id when available)
- `snapshot_id TEXT NOT NULL`
- `repo TEXT NULL`
- `commit TEXT NULL`
- `dataset_root_dir TEXT NOT NULL` (path used to build the views)
- `created_at TIMESTAMP NULL` (when the build DB was generated)
- `semantic_registry_path TEXT NULL` (if compiled as part of the run)
- `schema_manifest_path TEXT NULL`
- `buildspec_path TEXT NULL`

#### Table: `build_meta.datasets`

One row per dataset **in the expected inventory**, augmented with manifest-backed fields when the
dataset is present.

Rationale: build outputs can be missing due to failures, data quality, or feature staging. Downstream
services need to distinguish “expected but missing” from “unknown” without filesystem walks.

Suggested columns (minimal):

- `dataset_name TEXT NULL` (optional logical name, when available)
- `table_key TEXT NOT NULL` (e.g., `analytics.graph_metrics_functions`)
- `domain TEXT NOT NULL` (schema component of `table_key`, e.g., `analytics`)
- `table_name TEXT NOT NULL` (name component of `table_key`, e.g., `graph_metrics_functions`)
- `snapshot_id TEXT NOT NULL`
- `expected BOOLEAN NOT NULL` (True if in BuildSpec/contract inventory)
- `present BOOLEAN NOT NULL` (True if manifest exists and view was created)
- `dataset_dir TEXT NULL` (resolved snapshot directory when present)
- `manifest_path TEXT NULL` (path to `dataset_manifest.json` when present)
- `row_count BIGINT NULL` (manifest `row_count`)
- `schema_hash TEXT NULL` (manifest `schema_hash`)
- `partition_columns JSON NULL` (manifest `partition_columns`)
- `files JSON NULL` (manifest `files`; stored for introspection, not required for scanning)
- `contract_version TEXT NULL` (from `manifest.extras.contract_version`)
- `contract_hash TEXT NULL` (from `manifest.extras.contract_hash`)
- `settings_fingerprint TEXT NULL` (from manifest extras, if present)
- `created_at TEXT NULL` (manifest `created_at`, if present)

Notes

- Store `partition_columns` / `files` as JSON arrays for portability and easy inspection.
- Avoid duplicating full `table_schema` JSON unless you explicitly need it in SQL; it can be large.
  If you do store it, keep it in a separate table (see Optional tables).

#### Optional table: `build_meta.findings`

If you want the build-output DB to be the single “catalog + quality surface” for downstream
services, store run-level findings in SQL instead of (or in addition to) JSON/JSONL files.

Suggested columns:

- `table_key TEXT NULL`
- `severity TEXT NOT NULL` (e.g., `info`/`warn`/`error`)
- `check TEXT NOT NULL`
- `message TEXT NOT NULL`
- `details JSON NULL` (arbitrary structured payload)
- `recorded_at TIMESTAMP NULL`

#### Optional table: `build_meta.dataset_files`

Only needed if you want file-level inspection/joinability in SQL without parsing JSON.

- `table_key TEXT NOT NULL`
- `snapshot_id TEXT NOT NULL`
- `relative_path TEXT NOT NULL` (relative to `dataset_dir`)

#### Optional table: `build_meta.contracts`

Only needed if you want contract metadata in SQL beyond identity fields.

- `table_key TEXT NOT NULL`
- `contract_version TEXT NULL`
- `contract_hash TEXT NULL`
- `table_schema_json JSON NULL` (from manifest extras `table_schema`)
- `schema_drift_summary_json JSON NULL` (from manifest extras `schema_drift_summary`, if present)

### Data schemas: one per domain (e.g., `analytics`, `core`, `graph`, ...)

For each physical Parquet-backed dataset, create a **view** under its domain schema:

- `CREATE SCHEMA IF NOT EXISTS <domain>;`
- `CREATE VIEW <domain>.<table_name> AS SELECT ...`

This keeps the build DB’s query surface aligned with `table_key` and makes downstream attachment
intuitive (`build.<domain>.<table>`).

## View creation strategy (aligned with manifests + contracts)

### 1) Resolve scan inputs from the dataset manifest

Given `(dataset_root_dir, table_key, snapshot_id)`:

- Resolve `dataset_dir` using `dataset_snapshot_dir(...)` (`src/codeintel/core/datasets/paths.py`).
- Read `dataset_manifest.json`.
- Scan strategy (recommended): **always scan `dataset_dir`** (directory scan) and treat
  `manifest.files` as metadata only.
  - This matches the serving query planner’s Parquet scan behavior
    (`_parquet_scan_paths` in `src/codeintel/serving/semantic/duckdb_relation_builder.py`).
  - It avoids enormous view definitions for datasets with many files.
  - It tolerates partitioned layouts while still enabling `hive_partitioning`.

This mirrors the existing storage-side scan behavior in `src/codeintel/storage/gateway/accessors.py`.

### 2) Always enable `union_by_name`

Use `union_by_name=true` for Parquet scans to tolerate within-snapshot variation and avoid brittle
failures when columns appear/disappear across files.

This matches the existing scan adapter:

- `src/codeintel/serving/semantic/duckdb_scan_adapter.py` (`con.from_parquet(..., union_by_name=True)`).

### 3) Enable hive partitioning when manifests declare partitions

If `manifest.partition_columns` is non-empty, set `hive_partitioning=true` so DuckDB extracts
partition columns from paths.

### 4) Prefer view-only semantics

Views should scan Parquet in-place; do not `CREATE TABLE AS` into the build DB.

Downstream systems that need materialized tables should do so in their own DB (e.g., storage’s DB),
using DuckDB-to-DuckDB queries after attaching the build DB.

### 5) Attach contract identity as queryable metadata (not enforced by the view)

The build DB should store contract identity fields in `build_meta.datasets` (and optionally
`build_meta.contracts`) so downstream can:

- gate on `contract_hash` / `contract_version`
- detect unexpected drift between expected and produced outputs

Enforcement should remain in the build pipeline (alignment + validation) and/or in downstream
loading policy, rather than in view creation.

## Example view definitions (illustrative)

The exact SQL surface may vary by implementation, but the intent is:

- views scan Parquet paths
- scan options come from manifests

Example patterns:

- Scan a dataset directory:
  - `CREATE VIEW analytics.graph_metrics_functions AS`
    `SELECT * FROM parquet_scan('/abs/.../analytics/graph_metrics_functions/snapshot_id=.../',`
    `union_by_name=true, hive_partitioning=true);`

Notes

- Avoid emitting per-view file lists from `manifest.files` unless you have a strong correctness
  reason; for large datasets it can bloat view SQL and slow down attach/parse. Prefer directory
  scans with `union_by_name=true`.

## Downstream usage (DuckDB ↔ DuckDB)

Downstream (storage/serving) connects to its primary DuckDB DB, then attaches the build DB:

- `ATTACH '.../build_output.duckdb' AS build (READ_ONLY);`

Typical workflows:

- Query build outputs directly:
  - `SELECT COUNT(*) FROM build.analytics.graph_metrics_functions;`
- Materialize stable subsets into the storage DB:
  - `CREATE TABLE analytics.graph_metrics_functions AS`
    `SELECT * FROM build.analytics.graph_metrics_functions;`
- Validate/gate based on manifest metadata:
  - `SELECT table_key, contract_hash, row_count FROM build.build_meta.datasets;`

## Portability considerations

- **Absolute vs relative paths:** if you embed absolute paths in views, the build DB is tied to the
  filesystem layout. In practice, this is acceptable when the dataset root is mounted at a stable
  location in serving environments (serving already relies on absolute manifest paths).
- **Relocation:** DuckDB Parquet scan functions do not allow file paths to be looked up from tables
  at query time (no lateral column/subquery parameters), so a “dataset_root override” is not a
  reliable mechanism for persisted views. If paths change, **rebuild** the build-output DB.
- **Schema collisions:** always attach under a dedicated alias (e.g., `AS build`) and keep metadata
  under `build_meta` to avoid colliding with storage’s own catalogs.
- **Extension policy:** avoid designs that require extension auto-install at query time. Prefer
  core Parquet scanning with built-in DuckDB features.

## Minimal acceptance checks (for the build DB generator)

- Every row in `build_meta.datasets` with `present=true` corresponds to a queryable view under
  `<domain>.<table_name>`.
- Each view scan uses `union_by_name=true`.
- Hive partitioning flag matches `partition_columns` presence in the manifest.
- `contract_hash` / `contract_version` are present when the manifest extras include them.

## FastMCP Serving Delivery: Review + Build-DB Implications

The goal is to deliver CodeIntel data to an LLM programming agent via **FastMCP** under
`src/codeintel/serving`. Serving is already organized around a snapshot pointer + semantic
registry + schema inventory + export resources.

This section answers:

1) Given the current FastMCP implementation, what additional outputs (if any) should the
   build-output DuckDB provide?
2) What changes are implied for storage/serving so the system remains decoupled while enabling
   adaptive delivery to MCP clients?

### What FastMCP serving provides today (key surfaces)

The MCP server is assembled in `src/codeintel/serving/mcp/app.py` and registers:

- Tools (`src/codeintel/serving/mcp/tools/*`):
  - `semantic_catalog`: list semantic views
  - `semantic_describe`: schema + metadata (includes lineage when available)
  - `semantic_query`: safe query surface bounded by `view_id`
  - `semantic_explain`: compiled SQL + plan + derived lineage
  - `semantic_export`: materialize full results to an on-disk export store
  - `code_search`: BM25 search over `docs.search_documents` (requires FTS index)
  - `serving_meta`: server/snapshot metadata + limits/features
- Resources (`src/codeintel/serving/mcp/resources/*`):
  - `codeintel://semantic/views` and `codeintel://semantic/views/{view_id}`
  - `codeintel://meta/*` discovery + environment payloads
  - `codeintel://exports/*` for export payloads + metadata + previews + chunked retrieval
    (see `src/codeintel/serving/mcp/resource_store.py` and `src/codeintel/serving/mcp/resources/exports.py`)

Operationally, the serving layer is snapshot-driven:

- `current.json` points to a published snapshot (`src/codeintel/serving/db/pointer.py`).
- `ServingDBManager` loads the snapshot manifest + dataset manifests + semantic registry and
  hot-swaps connections (`src/codeintel/serving/db/manager.py`).
- `SemanticQueryKernel` executes semantic tools and uses:
  - the semantic registry (`semantic_registry.json`)
  - dataset manifests (Arrow dataset manifests) for scan planning + schema inventory
  - a DuckDB snapshot DB for metadata tables and index-backed features (FTS, lineage)
    (`src/codeintel/serving/semantic/kernel.py`)

### What is missing for “build-first, decoupled” agent delivery

FastMCP is currently optimized for *semantic views* (the curated, agent-friendly layer). In a
build-first world, the main gaps are:

- **Dataset-level introspection** (not just semantic views):
  - List all build dataset `table_key`s available for the snapshot.
  - Describe dataset manifest/contract identity (row count, contract hash/version, partitions,
    write settings) without requiring a semantic view to exist.
- **Diagnostics and drift visibility**:
  - Make contract identity + drift/alignment artifacts discoverable so an agent can explain “why a
    dataset looks weird” (missing columns, type coercions, etc.).
- **A safe “escape hatch” for non-semantic tables**:
  - There are valid agent workflows where a dataset exists but no semantic view is defined yet.
    Today, the MCP query surface is primarily `view_id`-bounded.

### What the build-output DuckDB should provide (additive)

The build-output DB is not strictly required for FastMCP to function: serving already loads dataset
manifests into memory (see `DatasetManifestIndex` usage in `src/codeintel/serving/db/manager.py`),
and the semantic query engine scans Parquet directly via manifest-aware planning
(`src/codeintel/serving/semantic/duckdb_relation_builder.py`).

However, to improve agent ergonomics and to keep “what exists in the snapshot” queryable without
filesystem walks, the build-output DB should include (in addition to the views themselves):

- `build_meta.datasets` should include the manifest-derived identity fields already listed above
  (contract hash/version, row_count, schema_hash, partition columns, files).
- Prefer including *optional* manifest extras that help debugging without opening Parquet:
  - `schema_drift_summary_json` (when present in manifest extras)
  - `write_settings` / `inferred_settings` (or a summary thereof)
- If you want to support dataset-level tools without reading many manifests at runtime, also emit
  `build_meta.dataset_files` (normalized file listing) as described earlier.

### Recommended FastMCP additions (serving-side)

To make the system “agent complete” while keeping the semantic layer as the primary interface,
add a **dataset discovery + inspection surface** parallel to semantic views:

1) **Resources (preferred for read-only discovery)**
- `codeintel://datasets` → dataset catalog (table keys + high-level metadata)
- `codeintel://datasets/{table_key}` → dataset description (manifest + contract identity + columns)
- `codeintel://datasets/{table_key}/manifest` → raw Arrow dataset manifest payload (verbatim)
- `codeintel://datasets/{table_key}/schema` → table schema payload (from manifest extras), if present

2) **Tools (only if you need query-like behavior)**
- `dataset_query` → a restricted query surface keyed by `table_key` (mirrors `semantic_query` but
  without requiring a semantic view).
  - This can reuse the existing scan + filter machinery already present for semantic execution:
    `DatasetManifestIndex` + `duckdb_relation_builder` + export pipeline.
  - Apply the same guardrails: timeouts, concurrency limiting, export-to-store with chunking.

These additions can be implemented without binding serving to the build-output DuckDB file:
serving already has the dataset manifests and can plan scans from them.

## Storage Redesign (Build-aligned) — What Should “Best-in-class” Look Like

`src/codeintel/storage` should be treated as the layer that turns **build-produced artifacts** into
an **immutable, serving-ready snapshot**, without re-coupling build execution to storage internals.

Given the current serving architecture, the minimal durable responsibilities for storage are:

1) **Own the serving snapshot DuckDB** (small, index/capability DB)
- Keep it read-only for serving, but writable during snapshot preparation.
- It must contain the tables/indexes required for serving-only capabilities that cannot live purely
  in Parquet (notably DuckDB FTS).

2) **Prepare snapshot-only derived artifacts**
- Build `docs.search_documents` + FTS index at snapshot time
  (`src/codeintel/storage/serving/search_index.py`,
  `src/codeintel/storage/serving/snapshot_service.py`).
- Ensure lineage metadata tables exist (currently validated by `ServingSnapshotService`):
  `metadata.derived_lineage_edges` and `metadata.derived_lineage_columns` under the meta catalog
  (`src/codeintel/storage/metadata/sync.py`, `src/codeintel/storage/serving/snapshot_service.py`).

3) **Publish and hot-swap snapshots**
- Continue to publish immutable snapshots + `current.json` pointer updates atomically
  (`src/codeintel/serving/publisher.py`, `src/codeintel/serving/db/manager.py`).

### Recommended storage boundary (inputs/outputs)

Storage should **not** be a dependency of build execution. Treat these as the only required inputs:

- Parquet datasets + Arrow dataset manifests (build outputs)
- `semantic_registry.json`, `schema_manifest.json`, `buildspec.json` (compiled build artifacts)

Storage output is a serving snapshot directory containing:

- `codeintel.duckdb` (the serving snapshot DB)
- `snapshot_manifest.json` (describes the snapshot and dataset manifests)
- copied build artifacts (`semantic_registry.json`, `schema_manifest.json`, `buildspec.json`,
  optional `environment.json`)
- `current.json` pointer update (atomic)

This is already the effective contract in `src/codeintel/serving/publisher.py`; the redesign goal is
to **rebuild storage internals** so they are strictly oriented around this publish contract and do
not assume long-lived, perfectly stable schemas during build development.

### How build-output DuckDB and storage DuckDB should interact

Use DuckDB-to-DuckDB attachment as an **integration convenience**, not a coupling point:

- Build-output DuckDB (`build_output.duckdb`) remains view-only and build-owned.
- Storage snapshot DB remains storage-owned and may materialize only what it needs (indexes and
  compact derived tables).

Two viable patterns:

- **Manifest-first (no dependency on build_output.duckdb):**
  - Storage reads dataset manifests, creates snapshot-local dataset views
    (current `ServingSnapshotService._register_dataset_views` does this).
  - Then builds search index and validates lineage.
  - This matches serving’s current behavior and keeps the boundary purely file-based.

- **Attach-first (optional convenience):**
  - Storage attaches `build_output.duckdb` into the snapshot DB under a dedicated alias
    (e.g., `ATTACH 'build_output.duckdb' AS build`).
  - Snapshot-local views can then be created as thin forwards (e.g., `CREATE VIEW core.modules AS
    SELECT * FROM build.core.modules`) if you want stable naming in the snapshot DB.
  - This is optional; it mainly reduces duplicated view-generation logic across layers.

### Serving changes implied by a storage rewrite (keep these minimal)

Serving is already snapshot-driven and should remain so. The changes that provide the most value
for agents, without destabilizing serving, are:

- Add dataset discovery resources/tools described above (so agents can operate even when semantic
  views lag behind build outputs).
- Extend `serving_meta` inventories/features to advertise dataset-level surfaces.
- Keep semantic execution unchanged: it should continue to rely on semantic registry + dataset
  manifests + snapshot DB capabilities (FTS, lineage).

If storage publishing adds new optional artifacts (e.g., `build_output.duckdb` in the snapshot
directory), expose them only through **structured MCP resources** (catalog/describe), not by
exposing filesystem paths to the agent.
