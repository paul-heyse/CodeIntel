# PyArrow + Polars Streaming Alignment Plan

## Purpose
Define a comprehensive implementation plan to adopt a streaming-first data flow in
`src/codeintel/build`, prioritize Arrow scanning, and minimize Arrow <-> Polars
interchange. This plan focuses on pyarrow and polars only.

## Goals
- Keep data in Arrow streams (RecordBatchReader, dataset scanners) end to end.
- Only convert to Polars LazyFrame when Polars expressions are required.
- Eliminate eager materialization in exports, validation, and dataset writes.
- Make streaming and scan-based access the default, with explicit opt-in to
  in-memory conversions.

## Non-Goals
- No changes to ingestion pipeline outside `src/codeintel/build`.
- No integration work for pandera, msgspec, or sqlglot in this plan.
- No changes to schema contracts beyond usage alignment.

## Key Principles
- Arrow is the canonical interchange format for data movement.
- Polars is a compute layer, not the default storage or transport format.
- Use scan-based access (pyarrow.dataset, polars scan) instead of read/collect.
- Prefer streaming writers (RecordBatchReader -> write) over frame conversions.

## Current Hot Spots (Streaming Breaks)
1) JSONL export and validation uses Polars conversions.
   - `src/codeintel/build/exports/writers.py`
   - `src/codeintel/build/exports/validation.py`
2) Analytics dataset insertion materializes Polars and converts to Arrow.
   - `src/codeintel/build/analytics/utilities/datasets.py`
3) Arrow dataset materialization uses `collect()` and `to_arrow()` in places.
   - `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`
4) Conversion utilities encourage eager DataFrame usage.
   - `src/codeintel/build/tabular/conversion.py`

## Implementation Phases

### Phase 1: JSONL export and validation streaming

#### 1.1 JSONL export: Arrow streaming writer
- Replace `pl.from_arrow(batch).write_ndjson` and `frame.write_json` with
  `pyarrow.json` streaming output.
- Preferred approach:
  - Use `pyarrow.json.write_json(reader, output_stream)` for full JSON output.
  - For JSONL, write batches via `pyarrow.json` or Arrow to Python generator
    and stream to file handle.
- Keep batch_size in `ExportRelation.fetch_record_batch`.

Targets:
- `src/codeintel/build/exports/writers.py`
- `src/codeintel/build/exports/jsonl.py`

Acceptance criteria:
- JSONL export does not convert Arrow batches to Polars.
- Output is line-delimited JSON objects as before.

#### 1.2 JSONL validation: Arrow JSON reader
- Replace `_read_jsonl_records` and `pa.Table.from_pylist` with
  `pyarrow.json.open_json` or `pyarrow.json.read_json` and return a
  `RecordBatchReader`.
- Feed the reader directly into `validate_record_batch_reader`.
- Preserve error reporting by collecting line-level errors from Arrow JSON
  decode exceptions when possible.

Targets:
- `src/codeintel/build/exports/validation.py`

Acceptance criteria:
- Validation uses `RecordBatchReader` as primary input.
- No full JSONL materialization into Python dicts.


### Phase 2: Analytics dataset inserts as Arrow streams

#### 2.1 Build Arrow batches from validated rows
- Replace `pl.from_dicts(normalized).to_arrow()` with chunked Arrow batch
  construction.
- Implement a small helper to convert rows to `RecordBatchReader` using
  `pa.RecordBatch.from_pylist` on fixed-size chunks.
- Pass the reader directly to `write_dataset`.

Targets:
- `src/codeintel/build/analytics/utilities/datasets.py`

Acceptance criteria:
- No Polars DataFrame required for dataset insertion.
- Arrow datasets are written from streaming readers.


### Phase 3: Arrow dataset materialization streaming-first

#### 3.1 Avoid eager `collect()` and `to_arrow()`
- Replace `_reader_from_frame` and call sites with a streaming reader path.
- Use `LazyFrameStream.to_reader()` where possible.
- For profiling or inspection, isolate any collect usage and ensure it only
  executes when the profile flag is enabled.

Targets:
- `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`

Acceptance criteria:
- Default materialization path does not call `collect()` or `to_arrow()`.
- Profiled path may collect, but is clearly separated and explicit.

#### 3.2 Use Polars streaming APIs when needed
- For Polars compute, use `collect_batches()` or `sink_batches()` to keep
  output streaming and avoid full DataFrame creation.
- Ensure any fallback path still uses Arrow readers and does not return
  a full DataFrame unless explicitly requested.


### Phase 4: Conversion utilities and API contracts

#### 4.1 Arrow-first conversion helpers
- Add explicit `arrow_reader_*` helpers for path-based and relation-based
  conversion without Polars.
- Make `tabular_to_frame` and `table_to_frame` call sites explicit and
  discourage their use for streaming flows.

Targets:
- `src/codeintel/build/tabular/conversion.py`
- `src/codeintel/build/tabular/frames.py`

Acceptance criteria:
- Internal callers in build prefer Arrow readers by default.
- Polars conversions are opt-in and documented as eager paths.


### Phase 5: Dataset scanning defaults

#### 5.1 Arrow scan as primary
- Favor `scan_snapshot_reader` over `scan_snapshot_lazyframe` in call sites.
- Add a small adapter to convert readers to LazyFrame only when required.

Targets:
- `src/codeintel/build/graphs/engine/datasets.py`
- Callers in `src/codeintel/build/graphs` and `src/codeintel/build/analytics`

Acceptance criteria:
- Scan + filter operations use Arrow dataset scanners by default.
- Polars is used when expression-based transforms are required.


## Cross-Cutting Design Notes
- Use `pyarrow.dataset.Scanner.to_reader()` for batch streaming.
- Consider `pa.unify_schemas` during multi-file scans (already available in
  Arrow datasets) when schema drift occurs.
- Prefer `RecordBatchReader` in internal interfaces to keep data streaming.

## Testing Strategy
- Add targeted tests for JSONL export and validation paths to ensure identical
  output semantics with Arrow-only writers.
- Add tests for `write_dataset` paths to validate streaming reader usage.
- For changes touching materialization, validate on a small dataset and
  compare manifest outputs.

## Risks and Mitigations
- Risk: JSONL output formatting differences between Polars and Arrow.
  Mitigation: golden file tests for JSONL output and strict schema validation.
- Risk: Arrow JSON reader error reporting is less granular than manual parsing.
  Mitigation: wrap exceptions with path and batch offsets where possible.
- Risk: Polars streaming APIs vary by version.
  Mitigation: gate usage behind feature checks (e.g., attribute existence).

## Rollout Plan
1) Land Phase 1 with focused tests on JSONL export/validation.
2) Land Phase 2 for analytics inserts and monitor dataset write performance.
3) Land Phase 3 and Phase 4 together to align materialization and conversions.
4) Land Phase 5 to shift scan defaults to Arrow readers.

## Acceptance Checklist
- All JSONL export and validation paths are Arrow streaming end to end.
- Analytics dataset inserts do not materialize Polars DataFrames by default.
- Materialization uses streaming readers unless explicitly profiling.
- Conversion helpers document eager paths and promote Arrow readers.
- Dataset scanning defaults to Arrow readers, with explicit Polars opt-in.

## References
- `docs/python_library_reference/pyarrow-advanced.md`
- `docs/python_library_reference/polars_advanced.md`
