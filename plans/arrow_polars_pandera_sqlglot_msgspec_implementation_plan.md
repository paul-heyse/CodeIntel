# Arrow/Polars/Pandera/SQLGlot/msgspec Implementation Plan (Detailed)

## Context
We want an aggressive migration to a streaming-first data plane with strict, typed contracts.
This plan focuses on consolidating CLI, storage, and core data flow around:
- PyArrow streaming (RecordBatchReader) as the canonical transport.
- Polars LazyFrame for transforms and streaming sinks.
- Pandera for contract enforcement at explicit stage boundaries.
- SQLGlot AST as the canonical query representation.
- msgspec.Struct for configuration and structured results.

Reference anchors:
- docs/python_library_reference/pyarrow-advanced.md
- docs/python_library_reference/polars_advanced.md
- docs/python_library_reference/pandera.md
- docs/python_library_reference/SQLGlot_advanced.md
- docs/python_library_reference/msgspec.md

## Goals
- Make Arrow streaming the internal data boundary across CLI, storage, and core.
- Consolidate CLI outputs into typed msgspec models and stream tabular output.
- Canonicalize SQL queries through SQLGlot AST, with strings only at storage boundaries.
- Standardize validation with Pandera (polars backend) and explicit validation profiles.
- Reduce duplicate parsing/serialization logic by centralizing in core utilities.

## Non-goals
- Full rewrite of all APIs at once.
- Immediate migration of every JSON column, but block new JSON columns in core.
- Changing external CLI surface unless it unlocks streaming or schema enforcement.

## Target Architecture (Data Plane)
- Core defines: schemas, streaming interfaces, query AST helpers, and validation profiles.
- Storage implements: DuckDB/Parquet IO and SQL rendering.
- CLI is a thin adapter: typed params in, streaming results out.

## Implementation Phases (with checklists)

### Phase 0: Baseline and guardrails
- [ ] Inventory all CLI handlers that return tabular data and note current materialization paths.
- [ ] Add or extend guardrails to discourage JSON in core and to flag full materialization in core.
- [ ] Decide a canonical CLI tabular result type (RecordBatchReader-first).

### Phase 1: msgspec foundations (CLI config + results)
- [ ] Convert CLI config models to msgspec.Struct with strict validation.
- [ ] Convert CLI result types to msgspec.Struct and remove custom to_dict logic.
- [ ] Centralize msgspec encode/decode (JSON and JSONL) in core serialization utilities.
- [ ] Update CLI rendering service to serialize msgspec results and stream JSONL.

### Phase 2: Columnar streaming surface for CLI
- [ ] Introduce a CLI tabular result type that wraps ColumnarStream or RecordBatchReader.
- [ ] Add rendering paths for:
  - [ ] JSON (materialize only when explicitly requested)
  - [ ] JSONL (stream batches)
  - [ ] Arrow IPC stream (optional flag)
- [ ] Ensure CLI handlers return streams for large datasets instead of lists.

### Phase 3: SQLGlot canonical query path
- [ ] Require SQLGlot AST for any CLI filter or query input.
- [ ] Normalize/optimize AST before storage compilation.
- [ ] Store AST payloads for diffing and reproducibility.

### Phase 4: Pandera validation gates
- [ ] Define validation profiles: schema-only, data-light, data-strict.
- [ ] Wire validation into storage and build boundaries.
- [ ] Ensure validation uses polars backend and accepts streaming inputs (LazyFrame/Arrow).

### Phase 5: Polars lazy and streaming sinks
- [ ] Replace eager DataFrame operations in core and build pipelines with LazyFrame.
- [ ] Use streaming sinks or collect_batches for exports and CLI.
- [ ] Add plan inspection output in verbose CLI modes (explain/profile).

### Phase 6: Migration cleanup
- [ ] Remove duplicate parsing/serialization logic in CLI handlers.
- [ ] Remove ad-hoc JSON path usage where Arrow/Polars is available.
- [ ] Standardize error/warning payloads to msgspec structs.

## File-by-File Change List (Detailed)

### New or Expanded Core Utilities
- [ ] src/codeintel/core/serialization/msgspec.py
  - Encode/decode helpers for JSON and JSONL.
  - Schema export via msgspec.json.schema_components.
- [ ] src/codeintel/core/columnar/stream.py
  - Extend to ensure RecordBatchReader-first APIs are available to CLI.
  - Add helpers to wrap Arrow Table and Relation into ColumnarStream.
- [ ] src/codeintel/core/columnar/ipc.py
  - Add Arrow IPC stream helpers for CLI output (if missing).
- [ ] src/codeintel/core/sqlglot_tools.py
  - Add canonical AST normalization and stable rendering utilities.

### CLI Core and Rendering
- [ ] src/codeintel/cli/core/result_types.py
  - Convert result models to msgspec.Struct.
  - Add a TabularResult type for columnar streams.
- [ ] src/codeintel/cli/core/results.py
  - Remove custom dataclass serialization.
  - Use msgspec to encode JSON/JSONL payloads.
- [ ] src/codeintel/cli/rendering/service.py
  - Add streaming output path for ColumnarStream.
  - Add JSONL streaming for large datasets.
- [ ] src/codeintel/cli/rendering/types.py
  - Add formats for jsonl and arrow IPC stream if not present.

### CLI Config and Param Parsing
- [ ] src/codeintel/cli/config/model.py
  - Replace dataclasses with msgspec.Struct or introduce a parallel msgspec model.
- [ ] src/codeintel/cli/config/loader.py
  - Replace manual parsing/coercion with msgspec.convert(strict=True).
- [ ] src/codeintel/cli/config/service.py
  - Use msgspec decoding for overrides and validation.

### CLI Handlers (streaming-first)
- [ ] src/codeintel/cli/handlers/storage.py
  - Return streaming results from storage queries.
  - Avoid list materialization for datasets and meta tables.
- [ ] src/codeintel/cli/handlers/datasets.py
  - Convert snapshot/diff output to msgspec structs.
  - Stream dataset listings as JSONL when requested.
- [ ] src/codeintel/cli/handlers/build_schema.py
  - Remove ad-hoc JSON parsing in favor of msgspec models.
  - Use Arrow schema + msgspec model output for manifests.
- [ ] src/codeintel/cli/handlers/graphs.py
  - Return tabular or structured results using msgspec only.

### Storage Query and Filter Pipeline
- [ ] src/codeintel/storage/queries/filter_compiler.py
  - Make SQLGlot AST the canonical representation.
  - Provide AST -> dialect SQL renderers only at storage boundary.
- [ ] src/codeintel/storage/queries/safe.py
  - Normalize and validate SQL AST before execution.
- [ ] src/codeintel/storage/views/diff.py
  - Store AST payloads and use sqlglot.diff for CLI diff output.

### Storage Data Plane (Arrow Streaming)
- [ ] src/codeintel/storage/datasets/arrow_store.py
  - Ensure APIs return RecordBatchReader for large scans.
- [ ] src/codeintel/storage/datasets/scanning.py
  - Prefer dataset Scanner.to_batches for streaming access.
- [ ] src/codeintel/storage/warehouse.py
  - Expose streaming query results (Arrow reader) as default.
- [ ] src/codeintel/storage/query_results.py
  - Add helpers for streaming JSONL without materialization.

### Build and Schema Services
- [ ] src/codeintel/build/schemas/inference_service.py
  - Ensure inference accepts RecordBatchReader and LazyFrame consistently.
- [ ] src/codeintel/build/hamilton/data_quality.py
  - Add explicit validation profiles and Pandera backend selection.
- [ ] src/codeintel/core/schemas/output_registry.py
  - Replace JSON columns with Arrow struct/map/list where needed.
- [ ] src/codeintel/core/schemas/generated_rows/*
  - Add msgspec Structs for row models and align Arrow schema metadata.

## Checklist by Workstream (Ready for Execution)

### Workstream A: CLI msgspec conversion
- [ ] Define msgspec Structs for CLI results.
- [ ] Update renderer to use msgspec JSON/JSONL encoding.
- [ ] Add tests for JSON and JSONL output stability.

### Workstream B: Streaming tabular results
- [ ] Add TabularResult wrapper for ColumnarStream.
- [ ] Implement JSONL streaming in renderer.
- [ ] Update handler return types to use TabularResult.

### Workstream C: SQLGlot AST canonicalization
- [ ] Parse CLI filter input into AST early.
- [ ] Normalize AST and store serialized AST where applicable.
- [ ] Render SQL only inside storage layer.

### Workstream D: Pandera validation profiles
- [ ] Define validation profiles and map to pipeline stages.
- [ ] Add Pandera schema validation at defined boundaries.
- [ ] Emit validation diagnostics as msgspec structs.

### Workstream E: Polars streaming transforms
- [ ] Replace eager DataFrame paths with LazyFrame.
- [ ] Use collect_batches or sink_* for streaming outputs.
- [ ] Add plan inspection at debug verbosity.

## Testing and Quality Gates
- [ ] Add unit tests for msgspec encode/decode of CLI results.
- [ ] Add integration tests for JSONL streaming output.
- [ ] Add query AST tests (parse -> normalize -> render).
- [ ] Add Pandera profile tests for schema-only vs data validation.
- [ ] Run tools.quality_report and fix all errors in touched files.

## Acceptance Criteria
- CLI returns structured outputs via msgspec and streams tabular data by default.
- SQL handling is AST-first, with rendering only at storage boundaries.
- Arrow streaming is the internal boundary across CLI and storage.
- Pandera validation gates are explicit and enforceable.
- No new JSON columns in core schemas; Arrow types are canonical.

## Rollout Notes
- Use feature flags where needed to avoid breaking CLI output contracts.
- Maintain temporary adapters for legacy JSON output during rollout.
- Update docs and CLI help text when output formats change.
