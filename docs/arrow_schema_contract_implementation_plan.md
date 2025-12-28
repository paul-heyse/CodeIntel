Arrow Schema Metadata as Contract - Best-in-Class Implementation Plan
=====================================================================

Goal
----
Make Arrow schema metadata the canonical, runtime contract while keeping the Hamilton DAG
as the build-time source of truth. The DuckDB meta store becomes a derived cache and diff
view, not the primary contract authority.

Best-in-Class Policy Decisions
------------------------------
1) Canonical source precedence
   - Hamilton DAG derived TableSchema is the root contract.
   - External ingest schemas are aligned to the contract; they do not mutate it.
   - DuckDB metadata is an authoritative cache only for validation and drift analysis.

2) Extras retention policy
   - Default policy: retain extras for external ingest, reject extras for Hamilton outputs.
   - Use a reserved internal column name (recommend "_ci_extras") to store extras as JSON.
   - Record extras schema drift as metadata (field name, inferred type, first/last seen).

3) Schema alignment rules
   - use pa.unify_schemas([contract, incoming]) only for alignment and diff.
   - "default" promote_options by default; "permissive" only for explicit opt-in tables.
   - Never auto-promote the canonical schema from unify output; promotion is explicit.

4) Arrow contract storage
   - Persist a serialized Arrow schema in metadata.schema_versions.renderer_cache.
   - Store base64 IPC schema bytes (serialize_schema) plus a contract version marker.
   - Keep TableSchema JSON as the stable human-readable source for diagnostics.

Current State Summary
---------------------
- Hamilton DAG first authority: schema_index + provider_unified + service build the
  schema provider chain and feed SchemaService.
- Inference uses Hamilton execution outputs and tabular annotations in
  build/schemas/inference_service.py and derives TableSchema via arrow_polars.py.
- Dataset contracts are built from SchemaService and persisted in DuckDB via
  build/meta/contract_catalog.py.
- Schema manifests are compiled from the build provider and persisted into
  metadata.schema_versions and metadata.table_schema_registry.
- Storage uses the contract catalog as its SchemaProvider and generates DDL from it.
- Arrow schema metadata is derived from TableSchema and enriched from metadata registry.

Target Architecture Overview
----------------------------
Build-time path (authoritative):
Hamilton DAG -> TableSchema -> Arrow Schema Contract -> Schema Manifest -> metadata registry

Runtime path (authoritative):
Arrow Schema Contract (from registry) -> IPC responses / validation / alignment

External ingest path:
Incoming schema -> align to contract -> retain extras -> write dataset -> drift report

DuckDB meta store role:
Derived cache and diff layer for schema drift, provenance, and audit history.

Contract Metadata Specification
-------------------------------
Schema-level metadata keys (Arrow Schema):
- "codeintel.schema_contract_version": "v1"
- "codeintel.table_key": "schema.table"
- "codeintel.schema_hash": "<hash>"
- "codeintel.schema_digest": "<digest>"
- "codeintel.primary_key": ["col1", "col2"]
- "codeintel.extras_policy": "retain|reject|drop"
- "codeintel.extras_column": "_ci_extras"
- "codeintel.extras_schema": { "field": "type", ... }
- "codeintel.provenance": { "derivation_kind": "...", "derivation_source": "...", ... }

Field-level metadata keys (Arrow Field):
- "codeintel.column_type": "VARCHAR|DOUBLE|..."
- "codeintel.nullable": true|false
- "codeintel.key_role": "primary_key|unique_index"
- "codeintel.description": "<optional>"
- "codeintel.provenance": { ... } (optional)
- "codeintel.lineage_edges": [ { "table_key": "...", "column": "..." } ]

Registry storage payload (metadata.schema_versions.renderer_cache):
- "arrow_schema_ipc_b64": "<base64 serialize_schema output>"
- "arrow_schema_contract_version": "v1"
- "arrow_schema_metadata": { "extras_policy": "...", ... } (optional convenience)

Implementation Plan (Phased)
----------------------------
Phase 0 - Policy and contract surface lock-in (no code)
1) Confirm reserved extras column name and policy defaults.
2) Confirm contract storage shape in renderer_cache (IPC base64).
3) Confirm unify_schemas usage rules and allowlist for permissive mode.
4) Confirm explicit denylist for extras retention (if any).
Acceptance criteria:
- Policy decisions documented in this plan and referenced from implementation docs.

Phase 1 - Canonical Arrow contract generation (build-time authority)
Files:
- src/codeintel/core/schemas/arrow_gen.py
- src/codeintel/core/schemas/arrow_polars.py
Steps:
1) Add contract version and extras policy metadata to arrow_schema_from_table_schema.
2) Add a helper like arrow_contract_for_table_schema(...) to centralize metadata keys.
3) Update arrow_polars.py to detect contract metadata and trust it when present.
Acceptance criteria:
- Arrow schema includes contract metadata keys for all tables.
- TableSchema derivation from Arrow validates contract version and metadata integrity.

Phase 2 - Persist Arrow contract into schema registry (sync path)
Files:
- src/codeintel/storage/tracking/schema_catalog_compile.py
- src/codeintel/storage/tracking/schema_catalog.py
- src/codeintel/build/hamilton/native/export/serving_artifacts.py
- src/codeintel/cli/handlers/meta.py
Steps:
1) Serialize Arrow schema to IPC bytes and store in renderer_cache.
2) Persist renderer_cache in schema_versions for every table and view.
3) Ensure serving_artifacts and meta.sync always include the contract payload.
Acceptance criteria:
- schema_versions.renderer_cache contains arrow_schema_ipc_b64 for all tables.
- No change in existing TableSchema JSON fields or schema hashes.

Phase 3 - Runtime consumption (storage + serving)
Files:
- src/codeintel/storage/schema/arrow_schema.py
- src/codeintel/core/exports/arrow_ipc.py
- src/codeintel/serving/http/streaming.py
Steps:
1) Load Arrow contract from schema_versions.renderer_cache when available.
2) Fall back to arrow_schema_from_table_schema only when contract is missing.
3) Ensure IPC responses use the contract schema and only append extra metadata.
Acceptance criteria:
- IPC output schema bytes match contract schema from registry.
- Runtime schema derivation no longer depends on duckdb registry metadata fields.

Phase 4 - External ingest alignment + extras retention
Files:
- src/codeintel/core/columnar/ (new module for schema_alignment.py)
- src/codeintel/build/hamilton/native/ingestion/frame_utils.py
- src/codeintel/build/hamilton/native/ingestion/ingest_targets.py
- src/codeintel/build/hamilton/native/ingestion/extraction_targets.py
Steps:
1) Implement align_reader_to_contract(reader, contract_schema, extras_policy, promote_options).
2) Build extras column as JSON object mapping extra fields to values.
3) Replace "unexpected columns" errors with extras retention for ingest sources.
4) Ensure Hamilton outputs still reject extras by default.
Acceptance criteria:
- External ingest retains extra fields in _ci_extras without failing.
- Hamilton outputs reject or flag extras, depending on policy.

Phase 5 - Meta store role change (cache and diff)
Files:
- src/codeintel/storage/gateway/factory.py
- src/codeintel/storage/tracking/schema_catalog.py
Steps:
1) Prefer runtime SchemaService if already configured; use metadata only for cache/diff.
2) Add drift report comparing Arrow contract metadata to registry metadata.
3) Emit warnings when metadata is stale or mismatched with contract schema hash.
Acceptance criteria:
- Storage opens without schema mismatches blocking runtime when contract exists.
- Drift report is visible and actionable, not silent.

Phase 6 - Backfill and migration
Files:
- src/codeintel/cli/handlers/meta.py
- src/codeintel/storage/metadata/schema.py (if new table is needed)
- src/codeintel/storage/metadata/ddl.py (if new table is needed)
Steps:
1) Add a backfill path to materialize Arrow contracts into renderer_cache.
2) If a new metadata table is added, extend DDL and migrate with meta.sync.
Acceptance criteria:
- Existing deployments can be upgraded with a single meta.sync run.

Phase 7 - Validation and testing
Files:
- tests for arrow_gen.py and arrow_polars.py
- tests for schema_catalog_compile.py and schema_catalog.py
- tests for ingestion alignment (new schema_alignment module)
Steps:
1) Round-trip: TableSchema -> Arrow contract -> TableSchema.
2) Alignment tests: extras retained and canonical columns preserved.
3) Registry tests: renderer_cache contains expected Arrow schema bytes.
Acceptance criteria:
- All tests pass; no regressions in existing schema validation paths.

Rollout Strategy
----------------
1) Implement Phase 1 and 2 only, behind a feature toggle (default on for new builds).
2) Backfill renderer_cache in metadata using meta.sync.
3) Switch runtime to prefer Arrow contract (Phase 3).
4) Enable external ingest alignment and extras retention (Phase 4).
5) De-emphasize DuckDB meta store as primary source (Phase 5).

Risk and Mitigation
-------------------
- Risk: Contract metadata drift from TableSchema JSON.
  Mitigation: strict hash match checks during persistence; alert on mismatch.
- Risk: Extras column collisions with source data.
  Mitigation: reserved name "_ci_extras" and explicit denylist for tables.
- Risk: Large renderer_cache payloads.
  Mitigation: store only schema (not data), use base64 IPC schema bytes.

Success Criteria
----------------
- Arrow contract metadata is the runtime authority for schema enforcement.
- Hamilton DAG remains the build-time source of truth.
- External ingest retains extras without data loss while preserving canonical schema.
- DuckDB meta store functions as a cache and diff layer, not a hard dependency.
