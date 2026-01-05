# Hamilton DAG Metadata Outputs

This document enumerates metadata generated during Hamilton DAG execution in
`src/codeintel/build`. It focuses on metadata artifacts, logs, and audit
records (not the primary dataset contents). Paths are relative to the runtime
`BuildPaths` values (build dir, dataset root, export dir).

## Storage roots and naming

- Build metadata bundle root: `{build_dir}/metadata`.
- Dataset snapshot root: `{dataset_root}/{snapshot_id}` where `snapshot_id` is
  `commit` if present, else `run_id`.
- Build log path: `{dataset_root}/{snapshot_id}/build_logs/build_{run_id}.jsonl`.
- Export artifact root: `{export_dir}`.
- Serving artifact root: `{build_dir}/serving/artifacts`.
- Decision trace artifact: `{build_dir}/decision_trace.json`.

## Build log (JSONL)

File: `{dataset_root}/{snapshot_id}/build_logs/build_{run_id}.jsonl`.

Each line is a JSON object with the base fields below, plus event-specific
payload fields.

Base fields:
- `event`
- `timestamp` (ISO-8601 UTC)
- `run_id`
- `repo`
- `commit`

Emitted events and fields (from `record_build_event` calls):
- `build.run.start`
  - `requested_targets_count`
- `build.run.complete`
  - `success`
  - `duration_ms`
  - `computed_targets_count`
  - `skipped_targets_count`
  - `failed_targets_count`
  - `error`
- `build.runtime.error`
  - `exception_type`
  - `error`
- `build.node.error`
  - `node_name`
  - `target`
  - `table_key`
  - `exception_type`
  - `error`
- `build.dataset.settings_fingerprint`
  - `table_key`
  - `target`
  - `settings_fingerprint`
- `build.schema.drift.detected`
  - `table_key`
  - `mode`
  - `details`
  - `missing_columns`
  - `extra_columns`
  - `type_changes`
- `build.schema.drift.blocked`
  - `table_key`
  - `mode`
  - `details`
- `build.inference.job.start`
  - `table_key`
  - `target`
  - `qparams_count`
  - `loader_overrides_count`
- `build.inference.job.ok`
  - `table_key`
  - `target`
  - `duration_ms`
- `build.inference.job.fail`
  - `table_key`
  - `target`
  - `exception_type`
  - `error`

## Cache manifest and decision trace

### Cache manifest entries (storage gateway)

Cache events are written to the storage gateway pipeline steps table via
`CacheManifestWriter` (module=`build`, stage=`cache`). These are not written
into the metadata bundle directly.

Fields recorded per cache event:
- `run_id`
- `module` ("build")
- `stage` ("cache")
- `name` (node name)
- `status` ("succeeded")
- `started_at`
- `completed_at`
- `row_counts` (always `null`)
- `extra`:
  - `cache_status` ("hit" | "miss" | "store")
  - `cache_key`
  - `cache_version`
  - `cache_path`
  - `duration_ms`
  - `size_bytes`
  - `target`

### Decision trace artifact

File: `{build_dir}/decision_trace.json`.

This is a JSON array of cache decision entries built from cache manifest
records. Each entry has:
- `index`
- `node_name`
- `target`
- `status` ("hit" | "miss" | "store")
- `cache_key`
- `cache_version`
- `data_version`
- `cache_path`
- `duration_ms`
- `size_bytes`
- `recorded_at` (ISO-8601)

## Materialization metadata (in-memory and downstream usage)

### MaterializationResult

Returned by Hamilton saver nodes and used to build target run records.
Fields:
- `status` ("succeeded" | "skipped" | "failed")
- `table_key`
- `row_count`
- `artifact_name`
- `path`
- `dataset_manifest_path`
- `size_bytes`
- `duration_ms`
- `input_hash`
- `error`

### TargetRunRecord

Used as the canonical per-target execution record, later written into run
reports and asset catalogs.

Fields:
- `target`
- `impl_kind` ("native")
- `status` ("succeeded" | "skipped" | "failed")
- `input_hash`
- `options_hash`
- `duration_ms`
- `row_counts`
- `error`
- `datasets` (tuple of `DatasetRef`)
- `artifacts` (tuple of `ArtifactRef`)
- `drift_summaries` (per-table schema drift summaries when available)

### DatasetRef

Fields:
- `table_key`
- `repo`
- `commit`
- `schema_version`
- `row_count`
- `source_target`
- `metadata`

### ArtifactRef

Fields:
- `name`
- `artifact_type`
- `repo`
- `commit`
- `path`
- `metadata`

## Dataset manifests and schema metadata

### Arrow dataset manifest

File: `{dataset_root}/{snapshot_id}/{table_key}/dataset_manifest.json`.

Fields (ArrowDatasetManifest):
- `dataset_id`
- `snapshot_id`
- `table_key`
- `partition_columns`
- `files`
- `schema_hash`
- `row_count`
- `stats`
- `created_at`
- `extras`

`extras` contents (from Arrow dataset saver):
- `table_schema` (TableSchema JSON)
- `provenance` (schema provenance, when available)
- `schema_drift_summary` (if drift detected)
- `settings_fingerprint`
- `inferred_settings` (if available)
- `write_settings` (materialization settings)

### Parquet schema metadata (embedded in Parquet files)

Written via Arrow dataset saver `schema_metadata` (parquet file metadata):
- `codeintel.table_key`
- `codeintel.domain`
- `codeintel.target`
- `codeintel.schema_hash`
- `codeintel.schema_digest`
- `codeintel.settings_fingerprint`
- `codeintel.columns_json`
- `codeintel.nullability_json`
- `codeintel.primary_keys_json`
- `codeintel.partition_columns_json`
- `codeintel.build_id`
- `codeintel.repo`
- `codeintel.commit`
- `codeintel.snapshot_id`
- `codeintel.generated_at`
- `codeintel.hamilton.node`
- `codeintel.hamilton.graph_version`
- `codeintel.inputs_json` (list of {`table_key`, `schema_hash`} for inputs)

### Schema observations (metadata bundle)

Files:
- `{build_dir}/metadata/schema/schema_versions.jsonl`
- `{build_dir}/metadata/schema/schema_observations.jsonl`

SchemaVersionRecord fields:
- `schema_digest`
- `schema_hash`
- `schema_json`
- `renderer_cache`
- `created_at`

SchemaObservationRecord fields:
- `observation_id`
- `table_key`
- `repo`
- `commit`
- `target_name`
- `schema_digest`
- `schema_hash`
- `arrow_schema_ipc_b64`
- `column_stats`
- `dataset_stats`
- `derived_settings`
- `drift_summary`
- `observed_at`

### TableSchema JSON shape

Used in schema manifests and dataset manifest extras:
- `schema`
- `name`
- `table_key`
- `description`
- `primary_key`
- `indexes`
- `columns`
- `write_policy` (optional)

Column JSON shape:
- `name`
- `type`
- `nullable`
- `description`

## Build metadata bundle (`{build_dir}/metadata`)

### bundle_manifest.json

Contains bundle-level metadata and per-file checksums:
- `bundle_schema_version`
- `generated_at`
- `repo`
- `commit`
- `run_id`
- `files` (list of {`path`, `sha256`, `size_bytes`, `record_count`, `schema_version`})

### contracts/contract_catalog.json + contracts/contract_catalog.hash

Contract catalog payload:
- `version`
- `contracts` (map of table_key -> DatasetContract payload)

DatasetContract payload fields:
- `table_key`
- `name`
- `schema` (TableSchema JSON)
- `json_schema_id`
- `jsonl_filename`
- `parquet_filename`
- `is_view`
- `owner_package`
- `tags`
- `description`
- `family`
- `owner`
- `freshness_sla`
- `retention_policy`
- `stable_id`
- `schema_version`
- `upstream_dependencies`
- `validation_profile`
- `composition` (composite schema details)

### schema/schema_manifest.json

SchemaManifest v2 payload:
- `version`
- `tables` (TableSchema JSON with provenance fields merged)
- `views` (TableSchema JSON with provenance fields merged)
- `artifacts` (ExportArtifact with provenance)

ExportArtifact fields:
- `kind` ("parquet" | "jsonl" | "json" | "csv")
- `filename`
- `table_key`
- `description`
- `provenance`:
  - `source_table_keys`
  - `source_schema_hashes`

### schema/schema_registry.json

Schema registry payload:
- `version`
- `generated_at`
- `repo`
- `commit`
- `entries` (TableSchemaRegistryRecord fields):
  - `table_key`
  - `schema_digest`
  - `schema_hash`
  - `derivation_kind`
  - `derivation_source`
  - `inference_status`
  - `inference_error`
  - `catalog_hash`
  - `updated_at`

### dataflow/dataset_nodes.jsonl

Record fields:
- `id` (table_key)
- `kind` ("table" | "view")
- `family`
- `owner_package`
- `description`

### dataflow/dataset_edges.jsonl

Record fields:
- `src` (table_key)
- `dst` (table_key)
- `edge_type` ("builds")

### lineage/derived_edges.jsonl

Record fields:
- `repo`
- `commit`
- `downstream`
- `upstream`
- `edge_type` ("derived_depends_on")
- `source` ("dag" | "view_lineage")
- `created_at`

### lineage/derived_columns.jsonl

Record fields:
- `repo`
- `commit`
- `downstream_table`
- `downstream_column`
- `upstream_table`
- `upstream_column`
- `edge_type` ("derived_column_depends_on")
- `source` ("view_lineage")
- `created_at`

### assets/asset_versions.jsonl

Record fields:
- `asset_kind` ("table" | "artifact")
- `asset_key`
- `version_hash`
- `schema_hash`
- `row_count`
- `bytes`
- `created_at`
- `meta` (fingerprint details, schema_hash, row_count, artifact_type, bytes)

### assets/asset_version_events.jsonl

Record fields:
- `run_id`
- `repo`
- `commit`
- `asset_kind`
- `asset_key`
- `version_hash`
- `status` ("materialized" | "reused")
- `target`
- `impl_kind`
- `location`
- `input_hash`
- `options_hash`
- `recorded_at`
- `meta`

### assets/run_asset_versions.jsonl

Record fields:
- `run_id`
- `repo`
- `commit`
- `asset_kind`
- `asset_key`
- `version_hash`
- `target`
- `resolution_kind` ("materialized" | "reused")
- `recorded_at`
- `meta`

### assets/asset_lineage.jsonl

Record fields:
- `downstream_kind`
- `downstream_key`
- `downstream_version`
- `upstream_kind`
- `upstream_key`
- `upstream_version`
- `edge_kind` ("depends_on")
- `created_at`
- `meta`

### exports/export_audit.jsonl

Record fields:
- `dataset`
- `macro`
- `rows`
- `duration_s`
- `output_path`
- `sql`
- `plan`
- `created_at`

### runs/run_report_{run_id}.jsonl

Record types:

1) `run_metadata`
- `run_id`
- `repo`
- `commit`
- `snapshot_id`
- `started_at`
- `duration_ms`
- `success`
- `computed_targets`
- `skipped_targets`
- `failed_targets`
- `error_summary`

2) `tag_schema_summary`
- `run_id`
- `repo`
- `commit`
- `snapshot_id`
- `summary` (tag schema summary + `spec_hash` + `spec_path`)

3) `output_catalog`
- `run_id`
- `repo`
- `commit`
- `snapshot_id`
- `output_kind` ("table" | "artifact")
- `table_key` (for tables)
- `artifact_name` (for artifacts)
- `artifact_type` (for artifacts)
- `artifact_path` (for artifacts)
- `target`
- `status`
- `row_count`
- `manifest_row_count`
- `schema_hash`
- `dataset_manifest_path`
- `output_role`
- `saver_node`
- `sink`
- `tags`

### runs/run_index.jsonl

Record fields:
- `run_id`
- `repo`
- `commit`
- `started_at`
- `duration_ms`
- `success`
- `report_path`
- `computed_targets_count`
- `skipped_targets_count`
- `failed_targets_count`

### tag schema file

File: `{dataset_root}/{snapshot_id}/tag_schema.json`.

Contents:
- `version`
- `primary_tags`
- `tag_keys` (list of {`key`, `value_type`, `allowed_values`})
- `required_tags` (table output + dataset node requirements)
- `allowed_values`

## Export metadata (document output)

### datasets_manifest.json

File: `{export_dir}/datasets_manifest.json`.

Payload:
- `datasets`: list of entries with:
  - `name` (dataset name)
  - `table` (fully qualified table/view key)
  - `jsonl` (filename, when present)
  - `parquet` (filename, when present)
  - `selected` (optional bool, when export is filtered)

### per-dataset manifest

File: `<artifact>.manifest.json` next to each export artifact.

Payload (ExportManifestData):
- `dataset`
- `validation_profile`
- `row_count`
- `data_hash`
- `started_at`
- `completed_at`
- `schema_id`
- `schema_version`
- `schema_digest`
- `artifact`
- `extras`

### incremental marker

File: `<artifact>.marker.json` next to each export artifact.

Payload:
- `dataset`
- `row_count`
- `schema_version`
- `validation_profile`
- `schema_digest`
- `exported_at`
- `extras`

## Serving artifacts (serving_artifacts target)

Files written under `{build_dir}/serving/artifacts`:
- `semantic_registry.json` (semantic registry v1)
- `schema_manifest.json` (SchemaManifest v2)
- `buildspec.json` (BuildSpec JSON)
- `dataset_manifest_paths.json`:
  - `dataset_manifest_paths` (list of existing dataset manifest paths)
- `environment.json`:
  - `generated_at`, `repo`, `commit`
  - `codeintel.version`
  - `python.version`, `python.implementation`
  - `platform.system`, `platform.release`, `platform.machine`
  - `tools.duckdb`, `tools.pyarrow`, `tools.sqlglot`
  - `duckdb.read_only`, `duckdb.extensions_env`
  - `duckdb.connect_env.threads`, `duckdb.connect_env.memory_limit`,
    `duckdb.connect_env.temp_directory`, `duckdb.connect_env.enable_profiling`,
    `duckdb.connect_env.profiling_output`
  - `argv0`

Tables produced as part of serving artifacts:
- `core.schema_inference_errors` (rows captured from schema inference failures)

### BuildSpec JSON shape

BuildSpec payload:
- `spec_version`
- `targets` (list of {`name`, `domain`, `impl_kind`, `deps`, `outputs`, `artifacts`})
- `datasets` (list of {`table_key`, `schema_hash`, `columns`})
- `semantic` ({`version`})
- `buildspec_hash`

## Notes on currently no-op persistence

The following hooks exist but do not write any payloads yet:
- `BuildRunWriter.start_run`
- `BuildRunWriter.complete_run`
- `BuildRunWriter.save_run_targets`
- `BuildRunWriter.save_run_nodes`

Node-level telemetry (`NodeExecutionRecord`) is collected during execution, but
`save_run_nodes` currently returns 0 and does not persist the records.
