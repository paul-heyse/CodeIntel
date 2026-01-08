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

3) **Dataset contracts / registry metadata** (optional, for name mapping)
- If you want `dataset_name → table_key` in the build DB, hydrate from contracts
  (`codeintel.build.schemas.iter_contracts`) at build-db creation time.

## File placement (recommended)

Because the DB’s views will reference snapshot Parquet paths, the build DB is most reliable when
co-located with (or deployed alongside) the snapshot root.

Recommended default:

- `dataset_root_dir/build_output.duckdb`

If you need the DB to live elsewhere, ensure the view definitions use absolute paths, or introduce a
runtime “dataset_root override” mechanism (see Portability considerations).

## Minimal DuckDB schema (tables + views)

### Schema: `build_meta`

This schema contains **small, persisted tables** that make the build DB self-describing.

#### Table: `build_meta.snapshot`

One row describing the snapshot this DB targets.

Suggested columns (minimal):

- `snapshot_id TEXT NOT NULL`
- `repo TEXT NULL`
- `commit TEXT NULL`
- `dataset_root_dir TEXT NOT NULL` (path used to build the views)
- `created_at TIMESTAMP NULL` (when the build DB was generated)

#### Table: `build_meta.datasets`

One row per dataset table key, derived from the Arrow dataset manifest.

Suggested columns (minimal):

- `table_key TEXT NOT NULL` (e.g., `analytics.graph_metrics_functions`)
- `domain TEXT NOT NULL` (schema component of `table_key`, e.g., `analytics`)
- `table_name TEXT NOT NULL` (name component of `table_key`, e.g., `graph_metrics_functions`)
- `snapshot_id TEXT NOT NULL`
- `dataset_dir TEXT NOT NULL` (resolved snapshot directory for this table)
- `manifest_path TEXT NOT NULL` (path to `dataset_manifest.json`)
- `row_count BIGINT NULL`
- `schema_hash TEXT NULL` (manifest `schema_hash`)
- `partition_columns JSON NULL` (manifest `partition_columns`)
- `files JSON NULL` (manifest `files`; optional, may be empty)
- `contract_version TEXT NULL` (from `manifest.extras.contract_version`)
- `contract_hash TEXT NULL` (from `manifest.extras.contract_hash`)
- `settings_fingerprint TEXT NULL` (from `manifest.extras.settings_fingerprint`, if present)
- `created_at TEXT NULL` (manifest `created_at`, if present)

Notes

- Store `partition_columns` / `files` as JSON arrays for portability and easy inspection.
- Avoid duplicating full `table_schema` JSON unless you explicitly need it in SQL; it can be large.
  If you do store it, keep it in a separate table (see Optional tables).

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
- Determine scan paths:
  - If `manifest.files` is non-empty: scan exactly those files (joined to `dataset_dir`).
  - Else: scan `dataset_dir` as a directory.

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
- Scan explicit files from a manifest:
  - `CREATE VIEW core.goids AS`
    `SELECT * FROM parquet_scan(['.../file1.parquet','.../file2.parquet'],`
    `union_by_name=true, hive_partitioning=false);`

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
  filesystem layout. If you need portability, add a “root override” convention and build views from
  a session variable (or rebuild the DB on the consumer side).
- **Schema collisions:** always attach under a dedicated alias (e.g., `AS build`) and keep metadata
  under `build_meta` to avoid colliding with storage’s own catalogs.
- **Extension policy:** avoid designs that require extension auto-install at query time. Prefer
  core Parquet scanning with built-in DuckDB features.

## Minimal acceptance checks (for the build DB generator)

- Every row in `build_meta.datasets` corresponds to a queryable view under `<domain>.<table_name>`.
- Each view scan uses `union_by_name=true`.
- Hive partitioning flag matches `partition_columns` presence in the manifest.
- `contract_hash` / `contract_version` are present when the manifest extras include them.

