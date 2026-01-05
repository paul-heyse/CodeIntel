# Polars/Arrow Re-architecture Implementation Plan (Arrow IPC Response Contract)

Status: design
Scope: build + serving + exports + validation

This plan defines a comprehensive, stepwise re-architecture toward a columnar-first
system with an Arrow IPC response contract. It focuses on service conversion,
technical de-risking, and validation gates. The end state makes `pa.RecordBatchReader`
and `pl.LazyFrame` the only in-process tabular contracts, and standardizes on Arrow IPC
streams for serving responses.

## Goals

- Establish an Arrow IPC response contract as the default serving format.
- Make `pa.RecordBatchReader` and `pl.LazyFrame` the only in-process table contracts.
- Eliminate row-tuple and pandas-centric pipelines from build and serving.
- Keep data lazy or streaming at boundaries; allow eager only as a last resort.
- Validate correctness with schema, type, and row-level invariants using Arrow/Polars.
- De-risk with targeted POCs and instrumentation before broad cutover.

## Non-goals

- Implement Flight or Flight SQL as the primary response format.
- Preserve pandas or row-tuple compatibility in build/serving paths.
- Replace DuckDB for complex queries (DuckDB remains optional as an extension engine).

## Arrow IPC Response Contract

### Contract definition

- Transport: Arrow IPC stream (not file format).
- Content type: `application/vnd.apache.arrow.stream`.
- Encoding: IPC stream with `IpcWriteOptions` defaults:
  - `compression="zstd"`
  - `metadata_version=V5`
  - `use_threads=True`
  - `unify_dictionaries=True`
- Schema metadata must include:
  - `codeintel.table_key`
  - `codeintel.schema_hash`
  - `codeintel.snapshot_id`
  - `codeintel.query_hash` (optional but recommended for traceability)
- Batch size: configurable (default aligned with `DEFAULT_ARROW_BATCH_SIZE`).
- Cancellation: reader iteration must check cancel tokens between batches.

### Contract boundaries

- Serving endpoints return Arrow IPC streams by default.
- JSONL remains an opt-in export format (not a serving default).
- Errors are reported via structured error responses (non-stream) with stable error codes.

## Canonical Columnar Interfaces

### ColumnarStream protocol

Define a single protocol type to unify conversion and streaming behavior:

- Required:
  - `schema: pa.Schema`
  - `to_reader(batch_size: int) -> pa.RecordBatchReader`
- Optional helpers:
  - `to_lazyframe() -> pl.LazyFrame`
  - `to_table() -> pa.Table` (last resort)

### Accepted in-process tabular types

- `pa.RecordBatchReader` (streaming boundary)
- `pl.LazyFrame` (lazy compute graph)

All other tabular forms are converted only at boundaries and only via explicit adapters.

## Phased Implementation Plan

### Phase 0: Contract and API scaffolding

Objective: Define the Arrow IPC contract and ColumnarStream interfaces.

Tasks:
- Add `ColumnarStream` protocol (new module, e.g., `src/codeintel/core/columnar/stream.py`).
- Add `ArrowIpcWriter` helper (IPC stream writer with metadata).
- Document response headers and schema metadata keys.
- Add integration spec doc to `docs/` (this document).

Deliverables:
- ColumnarStream protocol with adapters for `pa.RecordBatchReader` and `pl.LazyFrame`.
- Arrow IPC writer utility for serving and exports.
- Contract definition in docs.

Acceptance:
- No serving endpoint emits JSON when Arrow IPC is requested.
- IPC schema metadata includes the required fields.

### Phase 1: Streaming conversion utilities

Objective: Remove eager conversion calls at the conversion boundary.

Tasks:
- Replace `arrow_reader_to_lazyframe(reader.read_all())` with a streaming-friendly path:
  - Prefer `pl.from_arrow(reader)` if available for reader input.
  - Else: `pl.from_arrow(pa.Table.from_batches(reader))` only when explicitly requested.
- Replace `relation.arrow()` and `relation.fetch_arrow_table()` in conversion paths
  with `relation.fetch_arrow_reader()` and streaming conversion.
- Add explicit `to_table()` methods for rare eager use.

Target files:
- `src/codeintel/build/tabular/conversion.py`
- `src/codeintel/build/tabular/duckdb_relation.py`

Deliverables:
- Streaming-first conversion helpers.
- Explicit eager fallback paths with clear naming.

Acceptance:
- No `read_all()` in conversion helpers.
- No `relation.arrow()` for conversion.

### Phase 2: Ingestion outputs are columnar

Objective: Replace row-tuple outputs with columnar producers.

Tasks:
- Update ingestion tool outputs to emit:
  - `pl.LazyFrame` for transformation pipelines.
  - `pa.RecordBatchReader` for raw streaming payloads.
- Replace row-based dedupe with Polars expressions:
  - `frame.unique(subset=..., keep="last")`
  - Use `selectors` for schema-driven selection.
- Replace range-join row explosions:
  - Prefer `join_asof` or `join_where` where feasible.
  - Use Arrow `Table.join_asof` for pure Arrow paths.

Target files:
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`
- `src/codeintel/build/hamilton/native/ingestion/pipelines.py`

Deliverables:
- Ingestion steps return columnar payloads only.
- Removal of row tuple payloads and `save_rows` usage for ingestion targets.

Acceptance:
- No row tuples in ingestion targets.
- Dedupe and cleaning are expression-based.

### Phase 3: Dataset writes are streaming-first

Objective: Avoid full frame materialization in dataset writes.

Tasks:
- For `pl.LazyFrame`:
  - Prefer `sink_parquet` with partitioning if supported.
  - If partitioning requires eager materialization, use `collect_batches`
    and write via `ds.write_dataset`.
- For `pa.RecordBatchReader`:
  - Write directly via `ds.write_dataset`.
  - Capture row-group metadata and file stats.
- Populate dataset manifest stats from Parquet metadata:
  - row counts, row-group counts, sort keys, min/max stats.

Target files:
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
- `src/codeintel/storage/datasets/arrow_store.py`
- `src/codeintel/storage/datasets/manifests.py`

Deliverables:
- Streaming-first dataset writer with manifest stats.
- No unconditional `collect()` in dataset saver.

Acceptance:
- Partitioned writes do not force full collect.
- Manifests include row-group stats when available.

### Phase 4: Serving emits Arrow IPC streams

Objective: Standardize on Arrow IPC responses with cancellation.

Tasks:
- Update `PolarsExecutablePlan.to_reader()` to stream batches:
  - Use `LazyFrame.collect_batches()` or `sink_batches()` to build a reader.
  - Avoid `to_table()` for streaming response paths.
- Update semantic kernel to return IPC stream:
  - Use `pa.ipc.new_stream` over response handle.
  - Inject schema metadata before writing.
  - Respect cancellation between batches.
- Maintain DuckDB as optional engine, but emit IPC in all cases.

Target files:
- `src/codeintel/serving/semantic/engines/polars_engine.py`
- `src/codeintel/serving/semantic/kernel.py`
- `src/codeintel/serving/semantic/engines/duckdb_engine.py`
- `src/codeintel/build/exports/writers.py` (shared streaming writer utility)

Deliverables:
- IPC stream response for semantic queries.
- Cancellation support for streaming readers.

Acceptance:
- No `to_pylist()` or `to_pydict()` in serving paths.
- IPC responses are streamed with batch boundaries.

### Phase 5: Validation via Arrow/Polars (no pandas)

Objective: Replace pandera/pandas validation with Arrow/Polars checks.

Tasks:
- Define validation primitives on Arrow batches:
  - `Table.validate(full=False)`
  - `pyarrow.compute` expressions for constraints.
- Validate Parquet via `ParquetFile.iter_batches()`.
- Keep JSON Schema as contract, but evaluate constraints in Arrow/Polars.

Target files:
- `src/codeintel/build/exports/validation.py`
- `src/codeintel/storage/validation/pandera_df.py`
- `src/codeintel/build/schemas/constraints.py`
- `src/codeintel/core/schemas/row_models.py`

Deliverables:
- Arrow/Polars validation layer.
- Removal of pandas and pandera.pandas in build/serving paths.

Acceptance:
- No pandas usage in build/serving validation.
- Parquet validation is batch-based.

### Phase 6: Cleanup and removal of legacy paths

Objective: Remove row-tuple and pandas compatibility shims.

Tasks:
- Remove `DuckDBRowsSaver` and `save_rows` usage.
- Remove pandas fallback paths in serving result extraction.
- Update documentation and remove compatibility branches.

Acceptance:
- `rg -n "save_rows|DuckDBRowsSaver|pandas|to_pylist|to_pydict" src/codeintel/build src/codeintel/serving`
  returns no matches.

## De-risking and Validation Plan

### POCs (must complete before broad rollout)

- POC 1: Arrow IPC streaming response for a semantic query.
  - Assert schema metadata and batch sizes.
  - Validate cancellation mid-stream.
- POC 2: Streaming dataset write from `pl.LazyFrame` with partitioning.
  - Validate manifest stats derived from Parquet metadata.
- POC 3: Ingestion conversion of a single target (SCIP recommended).
  - Replace row tuples with columnar outputs end-to-end.

### Validation gates

- Functional:
  - Schema hashes remain stable across identical runs.
  - Row counts match pre-refactor baselines.
  - Query results are byte-for-byte identical after Arrow IPC decoding.
- Performance:
  - Peak RSS stays within budget (define per target).
  - Streaming responses begin within a defined latency threshold.
- Safety:
  - Cancellation stops reader iteration within N batches.
  - No `read_all()` or full `to_table()` in streaming paths.

### Observability

- Emit metrics:
  - batch counts, bytes, rows
  - IPC compression ratio
  - streaming time-to-first-batch
  - cancellation rate and cancel latency
- Add trace metadata:
  - `query_hash`, `snapshot_id`, `schema_hash`, `engine`

## Migration Strategy

- Stage by target: start with SCIP ingestion and one semantic view.
- Dual-run with comparison harness:
  - Old path executes in dry-run mode and results are compared.
  - Use `Table.equals` and hashing for diff detection.
- Cutover once parity is proven for the target group.

## Risks and Mitigations

- Polars streaming fallback for unsupported operators.
  - Mitigation: detect fallback via plan inspection and log warnings.
- Large partitioned writes forcing full collect.
  - Mitigation: prefer `collect_batches` -> `ds.write_dataset`.
- IPC client compatibility (non-Arrow consumers).
  - Mitigation: keep JSONL export as opt-in, not default serving.

## Testing Plan

- Unit tests:
  - ColumnarStream adapters and conversions.
  - IPC writer metadata injection.
- Integration tests:
  - Streaming serving response (round-trip decode).
  - Dataset manifest stats correctness.
  - Ingestion target conversion (SCIP).
- Regression:
  - Schema inference parity for Arrow/Polars outputs.

## Acceptance Checklist

- Arrow IPC stream is the default serving response.
- No pandas in build/serving paths.
- No row-tuple materialization in ingestion or materializers.
- Streaming conversions do not use `read_all()` or `to_pylist()`.
- Dataset manifests include row-group stats and schema hashes.

## Implementation Order (Recommended)

1. Phase 0: Contract and protocol scaffolding.
2. Phase 1: Conversion utilities (streaming-first).
3. Phase 4: Serving IPC streaming (core response contract).
4. Phase 2: Ingestion conversion (SCIP first).
5. Phase 3: Dataset writes streaming-first.
6. Phase 5: Validation rewrite.
7. Phase 6: Legacy removal.

## Quality Gates (Required)

- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted tests for build, serving, storage, and ingestion modules.

## Open Decisions

- IPC response framing over HTTP vs MCP: confirm output channel specifics.
- Batch size defaults for IPC and dataset writes.
- Manifest stats minimal set vs extended stats (min/max, distinct counts).

