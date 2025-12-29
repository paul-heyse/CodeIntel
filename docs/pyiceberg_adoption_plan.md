# PyIceberg Adoption Plan (Catalog-Centric, Hamilton-First)

## Intent

Adopt PyIceberg as the canonical metadata, schema, and snapshot layer while
preserving Hamilton DAG outputs as the source of truth for table definitions
and derived datasets. Replace bespoke dataset manifest handling with Iceberg
tables, snapshots, and metadata to harden schema evolution, query planning,
and streaming execution.

## Goals

- Make Iceberg table metadata the canonical contract plane derived from Hamilton DAG outputs.
- Centralize schema evolution via Iceberg `Schema` with stable field IDs.
- Replace dataset manifests with Iceberg snapshots and refs for serving.
- Enable streaming-first reads via `DataScan.to_arrow_batch_reader()`.
- Preserve Polars/DuckDB execution flexibility while improving pushdown.
- Improve observability and drift detection using Iceberg manifests/statistics.
- Track row-level deletes explicitly via tombstone/changelog tables.

## Non-goals

- Replacing Hamilton as the build-time authority for schema/targets.
- Introducing distributed compute engines beyond the current stack.
- Migrating all datasets in a single deployment window.

## Architectural Principles

- Hamilton DAG remains the primary schema authority; Iceberg schemas are derived artifacts.
- Iceberg catalog state is authoritative for storage/serving metadata and evolution.
- Serving snapshot identity is a ref/tag in Iceberg, not a separate manifest map.
- Streaming and lazy execution are defaults; eager materialization is explicit.

## Compatibility Notes (Hamilton + Serving/Storage Plans)

- SQLGlot remains the semantic query source of truth; Iceberg expressions are a projection.
- Arrow contract metadata stays derived from Hamilton schemas and aligns with Iceberg field IDs.
- Data-quality modifiers in Hamilton remain primary; Iceberg stats complement validation.
- No reintroduction of raw SQL templates or ad hoc view maps.

## Implementation Detailing Requirements (Applies to All Phases)

- Define entrypoints and outputs for each phase (public functions/classes + return types).
- Provide a per-phase acceptance checklist (unit, integration, and manual validation).
- Specify error semantics (fatal vs warning) and where errors surface (CLI, logs, metrics).
- Document performance expectations (streaming guarantees, memory caps, scan planning costs).
- Record guardrail exceptions and feature flags for incremental rollout.

## Chosen Implementation Decisions

- Catalog type: PyIceberg SQL catalog backed by DuckDB (single authoritative catalog DB).
- File IO: PyArrow FileIO with `pyarrow.fs.from_uri` for scheme-based resolution.
- Iceberg format: v2.
- Deletes: explicit tombstone/changelog tables (no Iceberg delete files initially).
- Snapshot refs: minimal scheme (`main`, `commit/<sha>`, `run/<run_id>`, `serving/<env>`).
- Metadata cache schema: `metadata.iceberg_*` (align with existing metadata namespace).
- Tombstone model: per-table `<schema>.<table>__tombstones` to preserve key typing and locality.
- Tombstone retention: at least the snapshot expiration window + a 7-day safety buffer.
- Metadata model: DuckDB advanced types (STRUCT/LIST/MAP/DECIMAL/TIMESTAMP_TZ);
  JSON retained only as fallback/audit fields.
- Query safety: SQLGlot AST building, canonicalization, and semantic diffing for metadata queries.

## Phase 0: Decisions and Guardrails

- Codify SQL-catalog-only policy (DuckDB file as the authoritative catalog).
- Codify Iceberg v2 and tombstone/changelog delete policy.
- Lock snapshot ref naming to the minimal scheme above.
- Add guardrails to block:
  - Non-Iceberg writes on migrated tables.
  - Equality delete operations.
  - Reads that bypass tombstone filtering where required.
- File targets: `tools/guardrails.py`, `src/codeintel/core/config/settings.py`.
- Acceptance: guardrails enforce Iceberg + tombstone-only behavior for opted tables.

Implementation detail to add:
- Guardrail rules list (exact checks + allowlist exceptions per table family).
- Error handling:
  - Guardrail violations are hard errors in build/serving.
  - Missing tombstone tables are warnings during phased rollout only.
- Feature flags:
  - `ICEBERG_WRITE_ENABLED`, `ICEBERG_READ_ENABLED`, `ICEBERG_TOMBSTONES_ENABLED`.
  - Precedence: env > config file > CLI args.
- Snapshot ref validation rules:
  - `serving/<env>` must exist to publish a snapshot.
  - `commit/<sha>` and `run/<run_id>` are immutable tags.

## Phase 1: PyIceberg Foundation Layer

- Introduce a small Iceberg adapter module:
  - `load_catalog` wrapper with `.pyiceberg.yaml` + env override handling.
  - `TableIdentifier` helper from table_key.
  - `Schema` conversion helpers between TableSchema and Iceberg Schema.
- Provision the DuckDB SQL catalog database (e.g., `iceberg_catalog.duckdb`).
- Attach the catalog DB as READ ONLY in serving/ops connections.
- Use parameterized SQL for catalog writes and Relation API for reads.
- Stream metadata reads with `fetch_arrow_reader()` for Arrow/Polars ingestion.
- Introduce a typed metadata cache layer in DuckDB for fast, interpretable querying.
- Add an Iceberg configuration section to CLI runtime settings.
- Provide a central `IcebergCatalogProvider` in `core` to keep access consistent.
- File targets:
  - `src/codeintel/core/iceberg/catalog.py` (new)
  - `src/codeintel/core/iceberg/schema.py` (new)
  - `src/codeintel/core/config/settings.py`
  - `src/codeintel/core/runtime/loader.py`
  - `src/codeintel/storage/iceberg/catalog_schema.py` (new)
- Acceptance: can load catalog + resolve table identifiers + convert schemas.

Implementation detail to add:
- `IcebergCatalogProvider` interface:
  - `load() -> Catalog`
  - `load_table(table_key: str) -> Table`
  - `table_exists(table_key: str) -> bool`
  - `resolve_identifier(table_key: str) -> tuple[str, ...]`
- Config sources and precedence:
  - `.pyiceberg.yaml` (default), env overrides, CLI overrides.
  - Required keys: `catalog.name`, `catalog.type`, `catalog.uri`.
- Catalog connection lifecycle:
  - Single writer connection for catalog updates.
  - Read-only attached connections for serving.
- Metadata cache refresh cadence:
  - On build completion and on serving snapshot load.
  - Explicit CLI command to force refresh.

## Catalog Data Model (DuckDB SQL + Advanced Types)

Maintain a compact, typed metadata cache for Iceberg tables in DuckDB. This is a
derived view of PyIceberg metadata intended for fast introspection and serving
planning. Use advanced types and keep JSON only as a fallback.

- `metadata.iceberg_tables`
  - `table_key VARCHAR`, `identifier VARCHAR`, `location VARCHAR`
  - `current_snapshot_id BIGINT`, `current_schema_id INT`
  - `current_spec_id INT`, `current_sort_order_id INT`
  - `properties MAP<VARCHAR, VARCHAR>`
  - `refs MAP<VARCHAR, STRUCT(snapshot_id BIGINT, ref_type VARCHAR, max_ref_age_ms BIGINT)>`
  - `last_updated_at TIMESTAMPTZ`
- `metadata.iceberg_schemas`
  - `table_key VARCHAR`, `schema_id INT`
  - `fields LIST<STRUCT(field_id INT, name VARCHAR, type VARCHAR, required BOOLEAN, doc VARCHAR, parent_id INT)>`
  - `schema_json JSON` (fallback)
- `metadata.iceberg_partition_specs`
  - `table_key VARCHAR`, `spec_id INT`
  - `fields LIST<STRUCT(field_id INT, name VARCHAR, transform VARCHAR, source_id INT)>`
- `metadata.iceberg_sort_orders`
  - `table_key VARCHAR`, `order_id INT`
  - `fields LIST<STRUCT(field_id INT, transform VARCHAR, direction VARCHAR, null_order VARCHAR)>`
- `metadata.iceberg_snapshots`
  - `table_key VARCHAR`, `snapshot_id BIGINT`, `parent_snapshot_id BIGINT`
  - `committed_at TIMESTAMPTZ`, `operation VARCHAR`
  - `summary MAP<VARCHAR, VARCHAR>`
  - `manifest_list_path VARCHAR`
- `metadata.iceberg_arrow_schema`
  - `table_key VARCHAR`, `schema_id INT`
  - `arrow_schema_ipc VARBINARY`
  - `arrow_schema_json JSON` (fallback)

SQLGlot should generate and canonicalize queries against this cache (AST build +
optimize + diff) to keep query fingerprints stable and safe.

DDL skeleton (DuckDB):

```sql
CREATE SCHEMA IF NOT EXISTS metadata;

CREATE TABLE IF NOT EXISTS metadata.iceberg_tables (
  table_key VARCHAR NOT NULL,
  identifier VARCHAR NOT NULL,
  location VARCHAR NOT NULL,
  current_snapshot_id BIGINT,
  current_schema_id INTEGER,
  current_spec_id INTEGER,
  current_sort_order_id INTEGER,
  properties MAP(VARCHAR, VARCHAR),
  refs MAP(VARCHAR, STRUCT(
    snapshot_id BIGINT,
    ref_type VARCHAR,
    max_ref_age_ms BIGINT
  )),
  last_updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS metadata.iceberg_schemas (
  table_key VARCHAR NOT NULL,
  schema_id INTEGER NOT NULL,
  fields LIST<STRUCT(
    field_id INTEGER,
    name VARCHAR,
    type VARCHAR,
    required BOOLEAN,
    doc VARCHAR,
    parent_id INTEGER
  )>,
  schema_json JSON
);

CREATE TABLE IF NOT EXISTS metadata.iceberg_partition_specs (
  table_key VARCHAR NOT NULL,
  spec_id INTEGER NOT NULL,
  fields LIST<STRUCT(
    field_id INTEGER,
    name VARCHAR,
    transform VARCHAR,
    source_id INTEGER
  )>
);

CREATE TABLE IF NOT EXISTS metadata.iceberg_sort_orders (
  table_key VARCHAR NOT NULL,
  order_id INTEGER NOT NULL,
  fields LIST<STRUCT(
    field_id INTEGER,
    transform VARCHAR,
    direction VARCHAR,
    null_order VARCHAR
  )>
);

CREATE TABLE IF NOT EXISTS metadata.iceberg_snapshots (
  table_key VARCHAR NOT NULL,
  snapshot_id BIGINT NOT NULL,
  parent_snapshot_id BIGINT,
  committed_at TIMESTAMPTZ NOT NULL,
  operation VARCHAR,
  summary MAP(VARCHAR, VARCHAR),
  manifest_list_path VARCHAR
);

CREATE TABLE IF NOT EXISTS metadata.iceberg_arrow_schema (
  table_key VARCHAR NOT NULL,
  schema_id INTEGER NOT NULL,
  arrow_schema_ipc VARBINARY,
  arrow_schema_json JSON
);
```

Indexes to add:
- `metadata.iceberg_*` tables: `table_key`, `schema_id`, `snapshot_id` as applicable.

## Phase 2: Schema Contract Alignment (Hamilton -> Iceberg)

- Convert Hamilton DAG-derived TableSchema to Iceberg Schema with stable field IDs.
- Store Iceberg schema JSON + field IDs in registry cache for drift comparison.
- Use `UpdateSchema.union_by_name` for permissive evolution when policy allows.
- Embed Arrow contract metadata with Iceberg field IDs and schema ID.
- File targets:
  - `src/codeintel/core/schemas/arrow_gen.py`
  - `src/codeintel/build/schemas/compile.py`
  - `src/codeintel/storage/tracking/schema_catalog.py`
  - `src/codeintel/storage/tracking/schema_catalog_models.py`
- Acceptance: schema registry includes Iceberg schema identifiers + metadata payloads.

Implementation detail to add:
- Field ID assignment:
  - Stable hash of `table_key + column_path`, truncated to int range.
  - Keep a persistent mapping for renamed columns to preserve IDs.
- Name mapping:
  - When a column is renamed, store old name in name mapping table.
  - Use name mapping during schema evolution to avoid ID churn.
- Schema evolution rules:
  - Default: strict (no type widening without explicit allowlist).
  - Per-table override: allow `union_by_name` for ingest-style tables.
- Arrow contract metadata:
  - Include `codeintel.iceberg_schema_id` and `codeintel.iceberg_field_id`.

## Phase 3: Build Write Path (Iceberg Snapshots + Tombstones)

- Replace Arrow dataset writes with Iceberg transactions:
  - `Table.append` for standard write policy.
  - `overwrite` or `dynamic_partition_overwrite` for replace policies.
  - `snapshot_properties` carry run_id, repo, commit, target_name, schema_hash.
- Use Iceberg partition spec and sort order derived from Hamilton tags.
- Preserve streaming behavior: convert `RecordBatchReader` to `pa.Table` in bounded chunks.
- Emit tombstones for deletions:
  - Create a companion Iceberg table `<schema>.<table>__tombstones`.
  - Mirror primary key columns and add:
    - `deleted_at TIMESTAMPTZ`
    - `snapshot_id BIGINT`
    - `run_id VARCHAR`, `commit VARCHAR`
    - `reason VARCHAR` (optional)
  - For full snapshots: diff previous snapshot vs current and append tombstones.
  - For incremental updates: append tombstones at delete time.
- File targets:
  - `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
  - `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`
  - `src/codeintel/core/execution/materialization.py`
- Acceptance: build outputs create Iceberg snapshots; manifests/metadata recorded.

Implementation detail to add:
- Write policy mapping:
  - Append-only: `Table.append`.
  - Replace-partition: `dynamic_partition_overwrite`.
  - Full replace: `overwrite` with `overwrite_filter=ALWAYS_TRUE`.
- Snapshot properties (required):
  - `run_id`, `commit`, `repo`, `table_key`, `schema_hash`, `producer_version`.
- Tombstone schema:
  - Primary key columns mirrored from base table.
  - Required: `deleted_at`, `snapshot_id`, `run_id`, `commit`.
  - Optional: `reason`, `source`.
- Incremental delete input:
  - Define a standard delete payload contract for ingestion nodes.
  - Validate delete keys against current schema before write.

## Tombstone Diff Algorithm (Full Snapshot Writes)

Use this algorithm when a table is rebuilt as a full snapshot and you need to
record deletes explicitly.

1) Identify the current and previous snapshot IDs.
   - `current_snapshot_id` is the new snapshot committed by the write.
   - `previous_snapshot_id` is the table snapshot before the write (if any).
2) Project only the primary key columns from both snapshots to minimize IO.
3) Compute deletes as the set difference:
   - `deleted_keys = previous_keys - current_keys`.
4) Emit tombstone rows for `deleted_keys`, tagged with:
   - `deleted_at` (commit timestamp), `snapshot_id` (current snapshot),
     `run_id`, `commit`, and optional `reason`.

Implementation patterns:
- **DuckDB streaming anti-join**
  - Scan both snapshots into Arrow RecordBatchReaders.
  - Register as DuckDB relations and `LEFT ANTI JOIN` on primary keys.
  - Stream the result into `<table>__tombstones` via Iceberg append.
- **Polars streaming anti-join**
  - `LazyFrame` from Arrow readers, then `join` with `how="anti"`.
  - `sink_batches` to append tombstone rows without materializing full results.

Incremental writes:
- If a pipeline already knows delete keys, skip the diff and append tombstones
  directly with the same metadata fields.
Retention:
- Purge tombstones only after the snapshot expiration window has elapsed,
  plus a 7-day buffer to avoid resurrecting deleted rows.

Implementation detail to add:
- Snapshot discovery:
  - Use table metadata to resolve the previous snapshot ID (if any) before write.
  - If no previous snapshot exists, skip diff and emit zero tombstones.
- Primary key typing:
  - Use Iceberg field IDs to map primary keys consistently across schema evolution.
  - Cast both sides to the canonical key types before diffing.
- Chunking/limits:
  - For large tables, stream PK projections and batch anti-join output.
  - Set a max tombstone batch size to keep memory usage bounded.

## Phase 4: Serving Read Path (Iceberg Scans)

- Replace dataset manifest reading with Iceberg scans:
  - `Table.scan(...).to_arrow_batch_reader()` for streaming IPC.
  - `Table.to_polars()` for lazy Polars plans (pushdown-aware).
  - `DataScan.to_duckdb()` for DuckDB interop when needed.
- Serving snapshot is resolved via Iceberg ref (tag/branch) instead of manifest paths.
- Apply tombstone filtering:
  - Inject `NOT EXISTS` or left anti-join against `<table>__tombstones`.
  - Filter tombstones to the current snapshot ref (`snapshot_id` or `run_id`).
  - Use SQLGlot AST transforms to apply this filter consistently.
- File targets:
  - `src/codeintel/serving/db/manager.py`
  - `src/codeintel/serving/semantic/kernel.py`
  - `src/codeintel/serving/semantic/engines/polars_engine.py`
  - `src/codeintel/serving/semantic/engines/duckdb_engine.py`
  - `src/codeintel/storage/serving/snapshot_service.py`
- Acceptance: serving reads only depend on Iceberg table metadata and refs.

Implementation detail to add:
- Read-path injection points:
  - Polars engine: build `LazyFrame` from `DataScan` and keep predicate pushdown.
  - DuckDB engine: use `DataScan.to_duckdb()` or register Arrow readers as relations.
- Snapshot resolution:
  - Resolve `serving/<env>` ref first, fallback to `main`.
  - Surface missing refs as warnings in serving (not fatal) unless a strict flag is set.
- Tombstone scoping:
  - Filter tombstones by `snapshot_id <= serving_snapshot_id`.
  - If `run_id` is available, prefer `run_id <= serving_run_id` for monotonic ordering.
- Missing tombstone table behavior:
  - Warn when tombstones are expected and not found.
  - Allow read path to proceed during rollout (`ICEBERG_TOMBSTONES_ENABLED=false`).
- Schema metadata:
  - Always attach `arrow_schema_ipc` metadata to responses for contract stability.

## SQLGlot Anti-Join Transform Pattern (Tombstone Filtering)

Inject a NOT EXISTS filter against `<table>__tombstones` using SQLGlot. This
keeps all filtering logic AST-based and avoids ad-hoc SQL string manipulation.

Example pattern (conceptual):

```python
from sqlglot import exp

def apply_tombstone_filter(select: exp.Select, *, table_key: str, pk_cols: list[str]) -> exp.Select:
    tombstone_table = f"{table_key}__tombstones"
    pk_predicates = [
        exp.EQ(
            this=exp.column(col, table=table_key),
            expression=exp.column(col, table=tombstone_table),
        )
        for col in pk_cols
    ]
    not_exists = exp.Not(
        this=exp.Exists(
            this=exp.select("1")
            .from_(tombstone_table)
            .where(exp.and_(*pk_predicates))
        )
    )
    return select.where(not_exists)
```

Notes:
- Bind `table_key` and tombstone table identifiers safely (no string interpolation
  into SQL text).
- Add snapshot scoping to the tombstone filter (e.g., `snapshot_id <= :snapshot_id`)
  as an extra predicate in the NOT EXISTS WHERE clause.
- Use `optimize(expr, schema=...)` before hashing/caching to canonicalize SQL.
- Ensure idempotency: detect and skip if an equivalent NOT EXISTS clause already exists.
- Canonicalize predicate order so query fingerprints are stable across runs.
- Include snapshot scoping predicates in the AST before canonicalization to ensure
  query hashes reflect the snapshot boundary.
- Use SQLGlot `diff` to compare pre/post-transform queries in guardrail checks.

## Phase 5: Observability + Validation via Iceberg Metadata

- Replace custom Parquet stats with Iceberg manifest metrics and statistics:
  - Use `table.inspect.entries()` and `table.inspect.manifests()` for stats.
  - Use `update_statistics` for derived stats persistence when appropriate.
- Align schema drift and validation reports to Iceberg schema IDs and snapshots.
- Add tombstone visibility:
  - Record delete counts by snapshot/run.
  - Expose tombstone summaries in schema observation payloads.
- File targets:
  - `src/codeintel/build/schemas/observations.py`
  - `src/codeintel/storage/validation/columnar.py`
  - `src/codeintel/storage/tracking/schema_catalog.py`
- Acceptance: drift/validation references Iceberg snapshot + schema IDs.

Implementation detail to add:
- Metrics (per table):
  - `iceberg_snapshot_count`, `iceberg_manifest_count`.
  - `iceberg_deleted_rows`, `iceberg_tombstone_rows`, `iceberg_tombstone_ratio`.
  - `iceberg_schema_version` and `iceberg_schema_drift` status.
- Drift classification:
  - `no_change`, `backward_compatible`, `breaking` (based on field ID + type).
- Observation payload:
  - Include `snapshot_id`, `schema_id`, `manifest_count`, `tombstone_count`.
- Validation policy:
  - Warnings only for missing tombstone coverage; hard error for schema mismatch.

## Phase 6: CLI + Operational Tooling

- Add CLI commands:
  - `iceberg.inspect` (snapshots, manifests, entries).
  - `iceberg.expire-snapshots` with retention policy.
  - `iceberg.time-travel` for read validation.
- Integrate `.pyiceberg.yaml` resolution into CLI runtime.
- File targets:
  - `src/codeintel/cli/commands/*`
  - `src/codeintel/cli/handlers/*`
- Acceptance: operators can inspect and maintain Iceberg metadata via CLI.

Implementation detail to add:
- Command specs:
  - `iceberg.inspect --table <key> --snapshots --manifests --entries`.
  - `iceberg.expire-snapshots --table <key> --retention-days <N> --dry-run`.
  - `iceberg.time-travel --table <key> --snapshot-id <id> --output <path>`.
- Output format:
  - JSON by default, `--format table` for human-readable output.
- Safety:
  - Require `--confirm` for destructive operations unless `--dry-run`.
  - Honor `ICEBERG_READ_ENABLED` for read-only deployments.

## Phase 7: Migration + Backfill Strategy

- Create a migration tool:
  - Read existing dataset manifests and import into Iceberg tables.
  - Backfill metadata location and create initial snapshots.
  - Store mapping from old manifest paths to new Iceberg refs.
- Backfill tombstones from diff between last two snapshots (where applicable).
- Run dual-write (Arrow dataset + Iceberg) for limited period if required.
- File targets:
  - `src/codeintel/storage/datasets/maintenance.py`
  - `src/codeintel/cli/handlers/meta.py`
  - `docs/storage_serving_best_in_class_plan.md`
- Acceptance: deterministic backfill, idempotent reruns, rollback plan documented.

Implementation detail to add:
- Per-table migration steps:
  1) Create Iceberg table with schema + partition spec derived from Hamilton.
  2) Write initial snapshot from latest Arrow dataset.
  3) Backfill metadata cache entries and Arrow schema IPC.
  4) Compute tombstones vs prior snapshot if it exists.
  5) Validate row counts + schema IDs + sampling checks.
- Idempotency:
  - Use snapshot tags to detect already-migrated tables.
  - If a tag exists, skip write and only refresh metadata cache.
- Rollback:
  - Flip feature flags to revert reads/writes to dataset manifests.
  - Keep Iceberg tables intact for reactivation.
- Dual-write acceptance:
  - Require identical row counts and schema ID alignment for N successive runs.

## Phase 8: Testing and Regression Gates

- Add tests for:
  - Schema evolution with field IDs and union_by_name.
  - Snapshot properties and time travel reads.
  - Scan planning vs streaming (batch reader vs eager).
  - Manifest count behavior (fast append vs merge).
- Add guardrail tests to block non-Iceberg writes for migrated tables.
- File targets:
  - `tests/build/*`
  - `tests/storage/*`
  - `tests/serving/*`
- Acceptance: contract tests enforce Iceberg-based behavior for migrated paths.

Implementation detail to add:
- Unit tests:
  - Field ID stability across rename + type preservation.
  - Snapshot property payload correctness.
  - Tombstone anti-join AST idempotency.
- Integration tests:
  - End-to-end write + read via Iceberg with tombstones applied.
  - Time-travel reads for `serving/<env>` refs.
- Performance checks:
  - Max memory for `DataScan.to_arrow_batch_reader()` under large scans.
  - Streaming batch size adherence in Polars/Arrow pipelines.

## Phase 9: Rollout and Cutover

- Gate migration by table family or dataset class.
- Provide feature flags for read/write switching:
  - `ICEBERG_WRITE_ENABLED`
  - `ICEBERG_READ_ENABLED`
  - `ICEBERG_REF_STRATEGY`
- Publish deprecation timeline for dataset manifests.
- Acceptance: gradual cutover with rollback path per table family.

Implementation detail to add:
- Rollout matrix:
  - Dev -> staging -> prod by table family.
  - Start with append-only tables, then replace/overwrite tables.
- Flag precedence:
  - Explicit CLI flags override env and config.
  - Enforce read/write symmetry in production (no read-only mismatch).
- Rollback plan:
  - Disable `ICEBERG_READ_ENABLED` and revert to manifests.
  - Keep Iceberg snapshots for post-incident analysis.

## Decision Log (fill during implementation)

- Catalog: DuckDB SQL catalog (`iceberg_catalog.duckdb`) + PyArrow FileIO.
- Format: Iceberg v2.
- Deletes: tombstone/changelog tables; no equality deletes.
- Snapshot refs: `main`, `commit/<sha>`, `run/<run_id>`, `serving/<env>`.
- Metadata cache schema: `metadata.iceberg_*`.
- Tombstone model: per-table `<schema>.<table>__tombstones`.
- Tombstone retention: snapshot expiration window + 7-day buffer.
- Metadata cache: STRUCT/LIST/MAP with JSON fallback for audit.
- SQLGlot: canonicalize metadata queries and enforce guardrail transforms.

## Success Metrics

- Zero bespoke dataset manifest reads for migrated tables.
- All serving reads use Iceberg `DataScan.to_arrow_batch_reader()`.
- Schema drift reports include Iceberg schema IDs and snapshot IDs.
- All migrated tables have tombstone coverage and enforced anti-join filtering.
- Reduced memory footprint for large exports and validations.
- Manifest and snapshot counts match configured maintenance policies.
