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
- [x] Decide a canonical CLI tabular result type (TabularResult wrapping ColumnarStream).

### Phase 1: msgspec foundations (CLI config + results)
- [x] Convert CLI config models to msgspec.Struct with strict validation.
- [x] Convert CLI result types to msgspec.Struct (include-none handling).
- [x] Remove legacy dataclass/to_dict serialization paths in CLI core.
- [x] Centralize msgspec encode/decode (JSON and JSONL) in core serialization utilities.
- [x] Update CLI rendering service to serialize msgspec results via centralized utilities.

### Phase 2: Columnar streaming surface for CLI
- [x] Add CLI columnar helpers for streaming tabular data (src/codeintel/cli/core/columnar.py).
- [x] Introduce a CLI tabular result type that wraps ColumnarStream or RecordBatchReader.
- [x] Add rendering paths for JSON and JSONL for ColumnarStream outputs.
- [x] Add Arrow IPC stream output format and renderer path.
- [ ] Ensure CLI handlers return streams for large datasets instead of lists.
  - PARTIAL: dataset list/jobs list/ops list/graph list/storage ingest logs converted.
- [ ] Default high-volume commands to JSONL output where streaming is supported.
- [ ] Add JSONL streaming tests for at least one storage and one dataset handler.

### Phase 3: SQLGlot canonical query path
- [ ] Require SQLGlot AST for any CLI filter or query input.
- [x] Normalize/optimize AST before storage compilation in safe query paths.
- [x] Store AST payloads for diffing and reproducibility.

### Phase 4: Pandera validation gates
- [x] Define validation profiles: schema-only, data-light, data-strict.
- [x] Wire validation profiles into Hamilton validation configuration.
- [ ] Wire validation into storage and build boundaries beyond Hamilton.
- [x] Ensure validation uses polars backend and accepts streaming inputs (LazyFrame/Arrow).

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
- [x] src/codeintel/core/serialization/msgspec.py
  - Encode/decode helpers for JSON and JSONL.
  - Schema export via msgspec.json.schema_components.
- [x] src/codeintel/core/columnar/stream.py
  - Extend to ensure RecordBatchReader-first APIs are available to CLI.
  - Add helpers to wrap Arrow Table and Relation into ColumnarStream.
- [x] src/codeintel/core/columnar/ipc.py
  - Add Arrow IPC stream helpers for CLI output (if missing).
- [x] src/codeintel/core/sqlglot_tools.py
  - Add canonical AST normalization and stable rendering utilities.
- [x] src/codeintel/core/validation/profiles.py
  - ValidationProfile enum and normalization helper.
- [x] src/codeintel/core/queries/safe.py
  - Canonicalize SQLGlot AST before execution (fallback to original on failure).
- [x] src/codeintel/storage/queries/safe.py
  - Canonicalize SQLGlot AST before execution (fallback to original on failure).

### CLI Core and Rendering
- [x] src/codeintel/cli/core/result_types.py
  - Convert result models to msgspec.Struct.
  - Add include-none handling for explicit null fields.
- [x] src/codeintel/cli/core/results.py
  - msgspec serialization is canonical; legacy dataclass serialization removed.
- [x] src/codeintel/cli/core/columnar.py
  - ColumnarStream helpers for CLI handlers.
- [x] src/codeintel/cli/rendering/service.py
  - Centralized msgspec JSON/JSONL encoding and Arrow IPC output path.
- [x] src/codeintel/cli/rendering/types.py
  - OutputFormat expanded to include Arrow IPC.

### CLI Config and Param Parsing
- [x] src/codeintel/cli/config/model.py
  - msgspec.Struct models with strict typing and output-format expansion.
- [x] src/codeintel/cli/config/loader.py
  - msgspec.convert(strict=True) with typed decode hooks.
- [ ] src/codeintel/cli/config/service.py
  - Use msgspec decoding for overrides and validation.

### CLI Handlers (streaming-first)
- [ ] src/codeintel/cli/handlers/storage.py
  - PARTIAL: ingest-cache-logs streams JSONL; other handlers still materialize.
- [ ] src/codeintel/cli/handlers/datasets.py
  - PARTIAL: list streaming; snapshot/diff output still materializes.
- [ ] src/codeintel/cli/handlers/build_schema.py
  - Remove ad-hoc JSON parsing in favor of msgspec models.
  - Use Arrow schema + msgspec model output for manifests.
- [ ] src/codeintel/cli/handlers/graphs.py
  - PARTIAL: list streaming; plan output still structured only.
- [ ] src/codeintel/cli/handlers/ops.py
  - PARTIAL: dataset list streaming; other ops still materialize.

### Storage Query and Filter Pipeline
- [x] src/codeintel/storage/queries/filter_compiler.py
  - SQLGlot AST canonicalization and safer compilation path.
- [x] src/codeintel/storage/views/diff.py
  - AST payloads included for diff/provenance output.

### Storage Data Plane (Arrow Streaming)
- [ ] src/codeintel/storage/datasets/arrow_store.py
  - Ensure APIs return RecordBatchReader for large scans.
- [ ] src/codeintel/storage/datasets/scanning.py
  - Prefer dataset Scanner.to_batches for streaming access.
- [ ] src/codeintel/storage/warehouse.py
  - Expose streaming query results (Arrow reader) as default.
- [x] src/codeintel/storage/query_results.py
  - Streaming JSONL helpers for Arrow readers and relations.

### Build and Schema Services
- [ ] src/codeintel/build/schemas/inference_service.py
  - Ensure inference accepts RecordBatchReader and LazyFrame consistently.
- [x] src/codeintel/build/hamilton/data_quality.py
  - Validation profiles applied in Hamilton quality checks.
- [ ] src/codeintel/core/schemas/output_registry.py
  - Replace JSON columns with Arrow struct/map/list where needed.
- [ ] src/codeintel/core/schemas/generated_rows/*
  - Add msgspec Structs for row models and align Arrow schema metadata.

## Checklist by Workstream (Updated)

### Workstream A: CLI msgspec conversion
- [x] Define msgspec Structs for CLI results.
- [x] Update renderer to use centralized msgspec JSON/JSONL encoding.
- [x] Convert CLI config models to msgspec.Struct with strict validation.
- [ ] Add tests for JSON and JSONL output stability.

### Workstream B: Streaming tabular results
- [x] Add ColumnarStream helpers for CLI handlers.
- [x] Add TabularResult wrapper for ColumnarStream or RecordBatchReader.
- [x] Implement JSONL streaming for ColumnarStream outputs.
- [ ] Update handler return types to use TabularResult.

### Workstream C: SQLGlot AST canonicalization
- [x] Canonicalize AST in safe query paths.
- [ ] Parse CLI filter input into AST early.
- [x] Normalize AST and store serialized AST where applicable.
- [ ] Render SQL only inside storage layer.

### Workstream D: Pandera validation profiles
- [x] Define validation profiles and map to Hamilton stages.
- [ ] Add Pandera schema validation at storage boundaries.
- [x] Emit validation diagnostics as msgspec structs.

### Workstream E: Polars streaming transforms
- [ ] Replace eager DataFrame paths with LazyFrame.
- [ ] Use collect_batches or sink_* for streaming outputs.
- [ ] Add plan inspection at debug verbosity.

## Outstanding Detailed Checklists

### Phase 0: Baseline and guardrails
- [ ] Build a handler inventory for src/codeintel/cli/handlers/*.py that records:
  - Current return types.
  - Whether data is materialized (list/dict) vs streamed (ColumnarStream).
  - Output format defaults and flags.
- [ ] Extend tools/guardrails.py to flag:
  - Core schema additions of JSON/VARIANT columns.
  - Full materialization of large Arrow datasets in core paths.
- [x] Decide canonical CLI tabular result type (TabularResult + ColumnarStream).
- [ ] Add a short doc note under plans/ that records the TabularResult decision.

### Phase 1: msgspec foundations (remaining)
- [ ] Audit src/codeintel/cli/config/service.py override flow (Cyclopts/TOML/env/CLI) and ensure
  msgspec conversion/validation is the canonical path for overrides.
- [ ] Add renderer stability tests for JSON and JSONL output (CLI results and TabularResult).

### Phase 2: Columnar streaming surface for CLI (remaining)
- [ ] Convert remaining handlers to TabularResult streaming outputs:
  - src/codeintel/cli/handlers/datasets.py (snapshot/diff)
  - src/codeintel/cli/handlers/build_schema.py (list/manifest output)
  - src/codeintel/cli/handlers/graphs.py (plan output)
  - src/codeintel/cli/handlers/ops.py (describe/verify outputs)
  - src/codeintel/cli/handlers/storage.py (non-ingest-cache outputs)
- [ ] Default high-volume commands to JSONL output:
  - codeintel datasets list
  - codeintel ops datasets list
  - codeintel graphs list
  - codeintel storage ingest-cache-logs (already JSONL, verify defaults)
- [ ] Add JSONL streaming tests:
  - One dataset handler (datasets list or snapshot)
  - One storage handler (ingest-cache-logs or storage list)

### Phase 3: SQLGlot canonical query path (remaining)
- [ ] Parse CLI filter input into SQLGlot AST early (before filter compilation) and pass AST
  through to storage query paths.
- [ ] Audit storage/query rendering to ensure SQL strings are generated only at storage
  boundaries (prefer render_sql_duckdb_safe and AST-based APIs).

### Phase 4: Pandera validation gates (remaining)
- [x] Storage columnar validation uses profile-aware context.
- [ ] Update src/codeintel/storage/schema/duckdb_contracts.py to propagate validation_profile
  into schema enforcement, where applicable.
- [ ] Update export validation to respect validation_profile (schema-only vs data-light/strict).
- [ ] Add diagnostics coverage for Pandera failures in storage/build logging (if not already).

### Phase 5: Polars lazy and streaming sinks
- [ ] Identify eager DataFrame usage (collect/to_pandas) in core/build and migrate to LazyFrame.
- [ ] Update exports to use streaming sinks or collect_batches where possible.
- [ ] Add explain/profile output in CLI verbose modes for Polars plans.

### Phase 6: Migration cleanup
- [ ] Remove duplicate parsing/serialization logic in CLI handlers in favor of core utilities.
- [ ] Remove ad-hoc JSON paths where Arrow/Polars are available in core and storage.
- [ ] Standardize error and warning payloads to msgspec structs across CLI and core.

## Testing and Quality Gates
- [ ] Add unit tests for msgspec encode/decode of CLI results.
- [ ] Add integration tests for JSONL streaming output.
- [x] Add query AST tests (parse -> normalize -> render).
- [x] Add Pandera profile tests for schema-only vs data validation.
- [ ] Run tools.quality_report and fix all errors in touched files.
  - Guardrails currently fail: duplicate Hamilton node `m__core__syntax_edges`.

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
