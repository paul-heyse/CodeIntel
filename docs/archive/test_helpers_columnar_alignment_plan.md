Test Helpers Columnar Alignment Plan
====================================

Goal
----
Align test helpers with the Arrow/Polars-first architecture by making Arrow
contracts and columnar streaming the default in test data generation,
materialization, validation, and goldens. This plan focuses on removing
row-tuple drift, reducing pandas usage, and enforcing schema contracts in
tests to mirror production behavior.

Guiding Principles
------------------
1) Arrow contract schema is the test truth.
2) Columnar streams (RecordBatchReader / LazyFrame) are the primary test
   data contract; tables are last resort.
3) No JSON stringification in helpers; keep JSON as dict/list and let Arrow
   or DuckDB handle encoding.
4) Test helpers should reuse production code paths to avoid drift.

Scope Overview
--------------
Target files and helper clusters:
- Columnar builders: `tests/_helpers/columnar_tables.py`
- Row protocol + builders: `tests/_helpers/builders/*.py`, `tests/_helpers/builders/row_protocol.py`
- Row factories: `tests/_helpers/fixtures/rows.py`
- Orchestration helpers: `tests/_helpers/orchestration/history.py`
- Snapshot factories: `tests/_helpers/serving_snapshot_factory.py`
- Fake storage: `tests/_helpers/fakes/storage.py`
- Golden helpers: `tests/_helpers/goldens/table_goldens.py`
- Dataset registry helpers: `tests/_helpers/dataset_factories.py`
- Validation assertions: `tests/_helpers/assertions/target_record_assertions.py`

Implementation Plan (Phased)
----------------------------

Phase 0 - Contract Alignment Baseline
1) Add a single contract-aware helper to resolve Arrow schemas for table keys:
   - Use `arrow_schema_for_table_key` when available, otherwise
     `arrow_contract_for_table_schema` via schema provider.
2) Establish a shared helper to enforce extras policy:
   - `align_reader_to_contract(reader, contract_schema, extras_policy=...)`.
Acceptance criteria:
- Helpers can resolve a contract schema for any table key used in tests.
- Helpers expose a single entry point for contract alignment.

Phase 1 - Columnar Stream Helper Layer (New)
Create a centralized helper module (suggested: `tests/_helpers/columnar_streams.py`) with:
- `reader_for_rows(table_key, rows, columns=None)` -> `pa.RecordBatchReader`.
- `lazyframe_for_rows(table_key, rows, columns=None)` -> `pl.LazyFrame`.
- `table_for_rows(table_key, rows, columns=None)` -> `pa.Table` (last resort).
Implementation details:
- Use `ColumnarRowBuffer`/`columnar_buffer_for_table_key` for row normalization.
- Apply Arrow contract metadata to the schema.
- Align readers with `align_reader_to_contract`.
Acceptance criteria:
- All test helpers can build readers/LazyFrames with contract metadata attached.
- No helper directly constructs `pa.table(...)` without contract metadata.

Phase 2 - Refactor Row Protocol + Builders to Columnar Writes
Update `tests/_helpers/builders/row_protocol.py` and row dataclasses:
- Replace tuple-only insertion with a columnar write path via Warehouse:
  - Build `RecordBatchReader` from row objects.
  - Use `Warehouse.materialize_table` with `ColumnarStream` or reader.
- Remove manual JSON stringification (use dict/list).
- Keep `InsertableRow.to_tuple()` for backwards compatibility but prefer
  `to_mapping()` for columnar ingestion.
Acceptance criteria:
- `insert_rows()` uses columnar materialization with contract alignment.
- Row builders do not emit JSON strings for JSON columns.

Phase 3 - RowFactory and Fixture Harmonization
Update `tests/_helpers/fixtures/rows.py`:
- Use schema service data directly for columns and types.
- Normalize values via `normalize_row_value_for_type`.
- Provide safe non-null defaults for non-nullable columns (configurable).
- Expose a columnar row builder that returns `ColumnarRows` or readers.
Acceptance criteria:
- All fixture rows are schema-aligned by type and nullability.
- Schema column ordering matches contract schema order.

Phase 4 - Replace Pandas with Arrow/Polars in Helpers
Update `tests/_helpers/orchestration/history.py`:
- Replace pandas DataFrames with Polars DataFrames or Arrow tables.
- Register data with DuckDB using Arrow/Polars interop.
Acceptance criteria:
- No pandas usage in history orchestration helpers.
- Tests continue to write/seed the same tables using columnar inputs.

Phase 5 - Snapshot and Dataset Helpers to Streaming
Update `tests/_helpers/serving_snapshot_factory.py` and dataset path helpers:
- Replace `fetch_arrow_table()` with `fetch_record_batch_reader()` or
  `fetch_arrow_table()` + `coerce_arrow_reader` to retain streaming.
- Use `ArrowDatasetWriteOptions` with contract schema or IPC schema metadata
  injected into the stream.
Acceptance criteria:
- Snapshot dataset writes follow streaming readers and carry contract metadata.
- Dataset manifests are generated with consistent schema_hash.

Phase 6 - Fake Storage Upgrade to Columnar
Update `tests/_helpers/fakes/storage.py`:
- Store data as `pa.Table` or `pa.RecordBatchReader` per table.
- Return schema-correct `RecordBatchReader` with contract metadata.
- Provide a minimal `ColumnarStream` adapter for tests that expect streaming.
Acceptance criteria:
- Fake storage returns Arrow readers with contract metadata.
- Unit tests can validate stream contracts without a DuckDB connection.

Phase 7 - Golden and Assertion Helpers
Update `tests/_helpers/goldens/table_goldens.py`:
- Dump tables using Arrow or Polars rather than tuple rows.
- Normalize via Arrow types (timestamps, JSON) for stable output.
Update `tests/_helpers/assertions/target_record_assertions.py`:
- Validate against record batch readers or tables with contract metadata.
Acceptance criteria:
- Golden output is stable and typed based on Arrow/Polars conventions.
- Assertions use columnar validation paths (`validate_record_batch_reader`).

Phase 8 - Dataset Registry Helpers
Update `tests/_helpers/dataset_factories.py`:
- Populate contract schemas using schema provider (no `schema=None`).
- Ensure dataset registry references the same schema as production.
Acceptance criteria:
- Test registries reflect production contract schemas.
- No helper constructs dataset contracts without schema definitions.

Phase 9 - Migration Strategy and Cleanup
1) Add deprecation notices in helper docstrings for tuple-based paths.
2) Migrate tests incrementally: high-traffic tests first, then legacy suites.
3) Remove unused row-tuple helpers after migration.
Acceptance criteria:
- No new tests rely on tuple-based helpers.
- Legacy tuple helpers are either removed or explicitly marked deprecated.

Validation Plan
---------------
Targeted test runs:
- `uv run pytest tests/_helpers -q`
- `uv run pytest tests/core/test_arrow_polars_schema.py tests/core/test_schema_alignment.py -q`
- `uv run pytest tests/storage/tracking/test_schema_catalog.py -q`
Contract validation:
- Use `validate_record_batch_reader` and `validate_table` on representative
  helpers to confirm alignment.

Open Design Decisions
---------------------
1) Preferred primary contract source for helpers:
   - `arrow_schema_for_table_key` via metadata registry vs direct TableSchema.
2) Default extras policy in tests:
   - Use contract policy from schema metadata or default to retain for
     ingest-style helpers.
3) Columnar output precedence:
   - Prefer Arrow `RecordBatchReader` vs Polars `LazyFrame` per helper type.

Deliverables
------------
- New `tests/_helpers/columnar_streams.py`.
- Updated helpers to use columnar streams with contract metadata.
- Reduced pandas usage and row-tuple drift.
- Updated goldens and assertions to use Arrow/Polars paths.

