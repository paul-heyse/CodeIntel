# Core Data Plane Migration Plan (Arrow + Polars + Pandera + SQLGlot + msgspec)

## Context
- We want an aggressive cutover to a cohesive, streaming-first data plane.
- `src/codeintel/core` must define the canonical data flow, contracts, and schema ownership.
- JSON is boundary-only (ingress/egress); internal flows are Arrow/Parquet.
- Use these references as policy anchors:
  - `docs/python_library_reference/pyarrow-advanced.md`
  - `docs/python_library_reference/polars_advanced.md`
  - `docs/python_library_reference/pandera.md`
  - `docs/python_library_reference/SQLGlot_advanced.md`
  - `docs/python_library_reference/msgspec.md`

## Goals
- Establish Arrow as the internal, streaming-native data plane.
- Adopt Polars lazy + streaming sinks for transforms.
- Use Pandera for contract enforcement at controlled stage boundaries.
- Make SQLGlot AST the canonical query representation.
- Replace JSON-centric core models with msgspec Structs.
- Reduce duplication and unstructured data flow in `src/codeintel/core`.

## Non-goals
- Rewriting all external APIs in one pass.
- Migrating every DuckDB JSON column immediately (but enforce no new JSON columns in core).
- Changing build/serving orchestration beyond data-plane boundaries.

## Guiding Principles
- Core owns schemas, contracts, and streaming interfaces.
- Storage/build/serving implement I/O and orchestration only.
- JSON is allowed only at ingress/egress; internal pipeline is Arrow/Parquet.
- Prefer streaming APIs that avoid full materialization.

## Target Architecture
```
core (Arrow contracts + streaming protocols + schema metadata + query AST)
  ^
  |
storage (DuckDB, Parquet/Arrow IO, SQL compilation)
  ^
  |
build/serving (orchestration, registry, runtime composition)
```

## Workstreams

### Workstream A: Core data-plane and guardrails
#### Completed
- [x] Hardened the JSON column guardrail regex in `tools/guardrails.py`.
- [x] Replaced direct JSON/orjson usage in core utilities with msgspec encoding
  (`src/codeintel/core/columnar/schema_alignment.py`,
  `src/codeintel/core/cache/keying.py`,
  `src/codeintel/core/hashing/fingerprint.py`).

#### Outstanding checklist
- [ ] Add an import-level guardrail to forbid `json`/`orjson` in `src/codeintel/core`,
  with an explicit allowlist for boundary modules (`core/helpers/json.py`,
  `core/exports/*`, `core/errors/problem_details.py`).
- [ ] Enforce “no JSON columns in core registries” by scanning
  `src/codeintel/core/schemas/output_registry.py` and
  `src/codeintel/core/schemas/table_registry.py`.
- [ ] Add a guardrail that forbids `Table.to_pylist()` in core (allow only
  exports/tests).
- [ ] Gate Arrow schema metadata validation (`schema_metadata_errors`) before
  materialization in storage/build materializers.
- [ ] Update `docs/core_data_format_policy.md` and add a documentation checkpoint
  (preflight or CI) to keep policy in sync.

### Workstream B: msgspec contract/model cutover
#### Completed
- [x] Added msgspec payload serialization in
  `src/codeintel/core/schemas/contract_serde.py`.
- [x] Updated `src/codeintel/build/meta/contract_catalog.py` to emit msgspec
  payloads.
- [x] Updated `src/codeintel/storage/contracts/provider.py` to decode payloads
  and use `msgspec.structs.replace`.
- [x] Added msgspec-based payload encode/decode with JSON fallback in
  `src/codeintel/core/helpers/payload.py` and
  `src/codeintel/core/helpers/json.py`.
- [x] Updated analytics read paths to decode BLOB payloads
  (`src/codeintel/build/hamilton/native/analytics/semantic_roles.py`,
  `src/codeintel/build/hamilton/native/analytics/entrypoints.py`,
  `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py`).

#### Outstanding checklist
- [ ] Convert `src/codeintel/core/manifests.py` to msgspec Structs and replace
  dataclass serializers.
- [ ] Convert `src/codeintel/core/schemas/contract_primitives.py` to msgspec
  Structs; remove dataclass-only helpers.
- [ ] Replace contract JSON schema export with msgspec JSON schema generation
  where appropriate.
- [ ] Introduce strict boundary decoders
  (`msgspec.json.Decoder(type=..., strict=True)`), and centralize ingress
  normalization to avoid JSON objects after decode.
- [ ] Add explicit legacy JSON normalization helpers for ingress-only paths.
- [ ] Update remaining call sites to use `contract_payload_*` helpers.
- [ ] Add msgspec round-trip tests for contracts/manifests and boundary decoders.

### Workstream C: Arrow schema and row model migration
#### Completed
- [x] Migrated core columnar fields to Arrow-native list/map/struct types for:
  `core.modules` tags/owners, `core.repo_map` modules/overlays,
  `core.ast_nodes` decorators, `core.cst_nodes` span/parents/qnames,
  `core.parse_manifest` future_imports, and `core.docstrings`
  params/returns/raises/examples.
- [x] Converted core syntax + tree-sitter extras payloads to Arrow-native structs
  and updated `core` generated row models.
- [x] Updated CST/tree-sitter ingestion and syntax augmentation to emit
  structured extras without msgpack encoding; encode only at export boundaries
  (e.g., CPG node extras).

#### Outstanding checklist
- [ ] Finish remaining JSON/BLOB replacements in core registries:
  - [ ] Audit `src/codeintel/core/schemas/output_registry.py` for any remaining
    JSON/BLOB columns in `core.*` tables.
  - [ ] Update `src/codeintel/core/schemas/table_registry.py` to mirror the
    Arrow-native types for those columns.
  - [ ] Update any affected ingestion/build producers and tests to emit
    structured values (no JSON objects).
- [ ] Regenerate/align generated row models where still byte-typed:
  - [ ] `src/codeintel/core/schemas/generated_rows/graph.py`
  - [ ] `src/codeintel/core/schemas/generated_rows/analytics.py`
- [ ] Add msgspec Struct row models for generated rows and adapters for
  typed row construction.
- [ ] Ensure Arrow schema metadata includes `codeintel.schema_hash`,
  `codeintel.schema_digest`, and dataset provenance keys.
- [ ] Add unit tests for Arrow schema metadata validation and row model
  alignment.

### Workstream D: Streaming-first Arrow IO
#### Outstanding checklist
- [ ] Audit core table materialization paths and replace table-wide operations
  with `RecordBatchReader`-based streaming readers.
- [ ] Replace eager dataset reads with `pyarrow.dataset.Scanner` streaming
  (`to_batches()` / `scan_batches()`).
- [ ] Adopt Arrow IPC streaming in core (`pyarrow.ipc.new_stream`) for
  in-memory transport.
- [ ] Switch file IO to `pyarrow.fs` streams; ensure no `Table.to_pylist` usage
  in core.
- [ ] Verify export-only modules still emit JSON/JSONL without leaking JSON into
  internal paths.

### Workstream E: Polars lazy pipeline adoption
#### Outstanding checklist
- [ ] Inventory core transforms using eager `DataFrame` and migrate to `LazyFrame`.
- [ ] Replace `read_*` with `scan_parquet`/`scan_ipc` in core IO nodes.
- [ ] Use streaming sinks (`LazyFrame.sink_parquet`) for outputs; reserve
  `collect_batches` for unavoidable Python batch logic.
- [ ] Add plan inspection in debug paths (physical plan, streaming engine).
- [ ] Add streaming plan tests to confirm non-materialized execution.

### Workstream F: Pandera contract enforcement
#### Outstanding checklist
- [ ] Define contract validation profiles (ingest, pre-materialize, export) and
  expected strictness per stage.
- [ ] Implement a core validation engine with `pandera.polars` adapters and
  explicit streaming-friendly entry points.
- [ ] Wire validation into dataset materialization boundaries (pre-write and
  export).
- [ ] Add CI checks that compile/validate Pandera schemas for all contracts.
- [ ] Add tests for schema-only vs schema+data validation behavior.

### Workstream G: SQLGlot canonicalization
#### Outstanding checklist
- [ ] Standardize AST-first query handling in `src/codeintel/core/queries`.
- [ ] Replace string-only query handling with SQLGlot AST operations
  (`parse_one`, `exp`, `transform`, `optimize`).
- [ ] Restrict SQL string rendering to storage boundaries only.
- [ ] Add query diff utilities for introspection and regression tests.

## Migration Sequencing

### Phase 0: Guardrails and policy
- [x] Harden JSON column guardrail regex in `tools/guardrails.py`.
- [ ] Add JSON import guardrails and `Table.to_pylist` guardrails in core.
- [ ] Update `docs/core_data_format_policy.md` to formalize Arrow-first rules.

### Phase 1: msgspec in core contracts
- [x] Switch contract payload serialization to msgspec.
- [ ] Convert manifests/contracts to msgspec Structs.
- [ ] Replace remaining JSON serialization helpers.
- [ ] Update call sites to use msgspec encoding/decoding consistently.

### Phase 2: Arrow schema and row models
- [x] Encode BLOB payloads in ingestion write paths.
- [x] Replace JSON columns with Arrow types for core modules/repo_map,
  AST/CST metadata, docstrings, syntax, and tree-sitter payloads.
- [ ] Finish remaining JSON/BLOB columns in core registries and update
  schema registries.
- [ ] Update generated row models for graph/analytics where still byte-typed.
- [ ] Backfill schema metadata validation.

### Phase 3: Streaming IO adoption
- Replace `to_table`/`to_pylist` paths with batch readers.
- Ensure Arrow IPC streaming is available in core utilities.

### Phase 4: Polars lazy pipelines
- Convert core data transforms to lazy plans.
- Introduce streaming sinks for outputs.

### Phase 5: Pandera validation enforcement
- Add schema-only vs data checks by stage.
- Wire validation into dataset materialization boundaries.

### Phase 6: SQLGlot canonicalization
- Convert core query compilation to AST-first.
- Restrict SQL string rendering to storage layer.

## Repository Impact Map (initial targets)

### Core
- `src/codeintel/core/manifests.py` (msgspec Structs)
- `src/codeintel/core/schemas/contract_primitives.py` (msgspec)
- `src/codeintel/core/schemas/contract_serde.py` (remove JSON-object serializers)
- `src/codeintel/core/schemas/output_registry.py` (Arrow types for nested fields)
- `src/codeintel/core/schemas/generated_rows/*` (typed structs)
- `src/codeintel/core/datasets/arrow_store.py` (streaming-first IO)
- `src/codeintel/core/columnar/schema_alignment.py` (no JSON normalization)
- `src/codeintel/core/queries/filter_compiler.py` (SQLGlot AST)

### Storage
- `src/codeintel/storage/queries/filter_compiler.py` (AST → SQL boundary)
- `src/codeintel/storage/schema/*` (Arrow metadata validation hooks)
- `src/codeintel/storage/datasets/*` (streaming IO compatibility)

### Build/Serving
- Use core contracts for schema + validation profiles.
- Ensure exports route through export-only JSON serializers.

## Acceptance Criteria
- Core has no JSON object pipelines except export/ingress boundary helpers.
- Core schemas contain no `JSON` column types unless export-only.
- All core dataset contracts have:
  - Arrow schema
  - Pandera schema
  - msgspec model
- Arrow schema metadata is validated before materialization.
- Internal pipelines use Arrow/Parquet or streaming readers.
- SQLGlot AST is canonical for query definitions in core.

## Testing Strategy
- Add unit tests for msgspec serialization round-trips.
- Add schema metadata validation tests on Arrow schemas.
- Add streaming IO tests using `RecordBatchReader` and `Scanner.to_batches`.
- Add Pandera profile tests (schema-only vs schema+data).
- Add SQLGlot AST translation tests with canonicalization.

## Risks and Mitigations
- Risk: Interface churn from aggressive refactor.
  - Mitigation: Temporary adapter layers in storage/build/serving.
- Risk: Performance regressions with new validation gates.
  - Mitigation: Stage profiles and sampling in Pandera.
- Risk: Streaming complexity and debugging difficulty.
  - Mitigation: Add instrumentation around batch processing and plan inspection.

## Deliverables Checklist
- [x] Hardened JSON column guardrail regex (`tools/guardrails.py`).
- [x] Msgspec payload serialization for contracts (`core/schemas/contract_serde.py`).
- [ ] Guardrails enforcing Arrow-first and JSON-boundary-only across core.
- [ ] msgspec-based manifests/contracts in core.
- [ ] Arrow-native row models and schema registries.
- [ ] Streaming-first dataset IO utilities.
- [ ] Polars lazy pipeline conversions.
- [ ] Pandera validation integration.
- [ ] SQLGlot AST canonicalization in core queries.
