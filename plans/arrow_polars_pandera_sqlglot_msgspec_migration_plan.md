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
1. Add guardrails to prevent JSON usage in core except for boundary modules.
   - Forbid `json`/`orjson` usage in `src/codeintel/core` except:
     - `src/codeintel/core/helpers/json.py`
     - `src/codeintel/core/exports/*`
     - `src/codeintel/core/errors/problem_details.py`
   - Forbid `Column("..._json", "JSON")` in core schema registries.
2. Add a guardrail that forbids `Table.to_pylist()` in core except exports/tests.
3. Promote Arrow schema metadata validation to an enforced gate.
   - Use `schema_metadata_errors` before materialization.
4. Add a core “data format policy” doc update checkpoint.

### Workstream B: msgspec contract/model cutover
1. Convert manifest and contract models to msgspec Structs.
   - Target modules:
     - `src/codeintel/core/manifests.py`
     - `src/codeintel/core/schemas/contract_primitives.py`
     - `src/codeintel/core/schemas/contract_serde.py`
   - Replace `to_json_obj` with msgspec serialization helpers.
2. Use msgspec JSON schema export for contract schemas.
   - Replace manual JSON schema generation where applicable.
3. Introduce msgspec-based decode/encode utilities for boundary parsing.
   - Prefer `msgspec.json.Decoder(type=..., strict=True)` for inbound.
4. Add conversion helpers to normalize legacy JSON payloads at ingress only.

### Workstream C: Arrow schema and row model migration
1. Replace JSON columns with Arrow `struct`, `map`, `list`.
   - Target schemas:
     - `src/codeintel/core/schemas/output_registry.py`
     - `src/codeintel/core/schemas/table_registry.py`
2. Replace `object` JSON row fields with structured types.
   - Target models:
     - `src/codeintel/core/schemas/generated_rows/core.py`
     - `src/codeintel/core/schemas/generated_rows/graph.py`
     - `src/codeintel/core/schemas/generated_rows/analytics.py`
3. Add row model Structs (msgspec) for generated rows.
4. Ensure Arrow schema metadata includes:
   - `codeintel.schema_hash`
   - `codeintel.schema_digest`
   - Dataset provenance keys used in schema alignment.

### Workstream D: Streaming-first Arrow IO
1. Replace internal table materialization with stream readers.
   - Prefer `pyarrow.RecordBatchReader.from_stream`.
2. Use dataset scanners for streaming reads.
   - `pyarrow.dataset.Scanner.to_batches()` or `.scan_batches()`.
3. Replace table-wide IO in core with:
   - `pyarrow.ipc.new_stream` for in-memory streaming transport.
   - `pyarrow.fs` input/output streams for IO.
4. Ensure export-only modules can still emit JSON/JSONL.

### Workstream E: Polars lazy pipeline adoption
1. Replace eager `DataFrame` operations in core with `LazyFrame`.
2. Use `scan_parquet`/`scan_ipc` for IO nodes, not `read_*`.
3. Use streaming sinks:
   - `LazyFrame.sink_parquet` for outputs.
   - `collect_batches` only when Python-level batch logic is required.
4. Add plan inspection gates in debug paths:
   - `.show_graph(plan_stage="physical", engine="streaming")` where useful.

### Workstream F: Pandera contract enforcement
1. Add contract validation profiles per stage:
   - Ingest: schema-only
   - Pre-materialize: schema + data
   - Export: schema + data (strict)
2. Implement a core validation engine with backend adapters:
   - `pandera.polars` as the primary backend.
3. Add contract compile checks to CI for all registered schemas.

### Workstream G: SQLGlot canonicalization
1. Standardize query AST usage in `src/codeintel/core/queries`.
2. Replace string-only query handling with AST-first:
   - `parse_one`, `exp`, `transform`, `optimize`.
3. Make SQL string rendering a storage-only boundary.
4. Add query diff utilities for introspection (optional).

## Migration Sequencing

### Phase 0: Guardrails and policy
- Add guardrails in `tools/guardrails.py`.
- Update `docs/core_data_format_policy.md` to formalize Arrow-first rules.

### Phase 1: msgspec in core contracts
- Convert manifests/contracts to msgspec.
- Replace JSON serialization helpers.
- Update call sites to use msgspec encoding/decoding.

### Phase 2: Arrow schema and row models
- Replace JSON columns with Arrow types.
- Update generated row models and schema registries.
- Backfill schema metadata validation.

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
- Guardrails enforcing Arrow-first and JSON-boundary-only.
- msgspec-based manifests/contracts in core.
- Arrow-native row models and schema registries.
- Streaming-first dataset IO utilities.
- Polars lazy pipeline conversions.
- Pandera validation integration.
- SQLGlot AST canonicalization in core queries.
