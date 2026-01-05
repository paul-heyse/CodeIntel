Pytest Structure Refactor Plan (Arrow/Polars Contract-First)
===========================================================

Goal
----
Refactor pytest structure so tests mirror the Arrow/Polars-first architecture,
prioritizing contract schemas, columnar streams, and streaming-safe validation
paths. The plan replaces tuple/row and pandas-centric patterns with reusable
columnar fixtures and assertions that align with production behavior.

Drivers
-------
- Arrow schema metadata is now the contract; tests must treat it as the source
  of truth, not a derived artifact.
- In-process contracts are columnar (RecordBatchReader, LazyFrame), with tables
  as a last resort.
- Streaming is a first-class runtime mode; tests should avoid read_all() or
  fetchall() assumptions.

Design Principles
-----------------
1) Contract-first: resolve schema via SchemaService or Arrow contract metadata
   for every test table.
2) Columnar-first: prefer RecordBatchReader or LazyFrame for inputs and outputs.
3) Streaming-safe: validate via readers and batches; materialize only at edges.
4) Production parity: reuse production helpers for schema alignment and
   normalization to avoid drift.
5) No JSON stringification: keep JSON as dict/list and let Arrow/DuckDB encode.

Scope
-----
In scope:
- Shared fixtures and factories in tests/_helpers and tests/conftest.py.
- Test ingestion helpers, assertions, and golden snapshot paths.
- Replacement of tuple-based test inputs and pandas-only helpers.
Out of scope:
- Product code changes not required for tests to align.
- One-off test rewrites unrelated to schema/columnar architecture.

Implementation Plan (Phased)
----------------------------

Phase 0 - Inventory and Baseline
1) Catalog test helper entry points and current tuple/pandas usage.
2) Identify test suites that still depend on row tuples or fetchall().
3) Define baseline contract-resolving helper API.
Acceptance criteria:
- A list of affected helpers and top 10 test suites by usage.
- A minimal contract fixture API agreed on.

Phase 1 - Contract Fixture Layer (Pytest Structure)
1) Add a session-scoped SchemaService fixture and contract resolver:
   - contract_schema_for(table_key) -> pa.Schema
2) Add a columnar factory fixture:
   - columnar_rows_for(table_key, rows) -> ColumnarRows
   - reader_for_rows(table_key, rows) -> RecordBatchReader
3) Centralize extras policy and alignment:
   - align_reader_to_contract(reader, contract_schema, extras_policy)
Acceptance criteria:
- Any test can request a contract schema and a reader for a table key.
- Extras policy is configured once and applied consistently.

Phase 2 - Data Generation and Normalization
1) Replace tuple-only row helpers with mapping-based helpers:
   - to_mapping() becomes primary; to_tuple() stays for temporary compat.
2) Normalize all fixture values via normalize_row_value_for_type.
3) Convert JSON string fields in fixtures to dict/list.
Acceptance criteria:
- All test helper outputs are contract-aligned by type and nullability.
- No helper emits JSON strings for JSON columns.

Phase 3 - Streaming-Safe Assertions and Query Helpers
1) Replace fetchall() assertions with fetch_record_batch() or readers.
2) Validate with validate_record_batch_reader and schema checks.
3) Provide a table materialization helper only at the edge:
   - table_for_rows(table_key, rows) for compatibility.
Acceptance criteria:
- Tests validate streaming outputs without read_all().
- Assertions operate on readers or Arrow tables with contract metadata.

Phase 4 - Golden Snapshots and IPC Baseline
1) Standardize golden snapshots on Arrow IPC payloads.
2) Store schema metadata alongside IPC for contract drift detection.
3) Provide a helper to compare IPC schema metadata and row content.
Acceptance criteria:
- Golden tests consume IPC by default.
- Contract metadata comparison is available for snapshot tests.

Phase 5 - DuckDB and External Integration Helpers
1) Ensure DuckDB helpers accept Arrow readers or Polars LazyFrames.
2) Use register-from-reader and fetch_record_batch() consistently.
3) Remove pandas-only pathways where parity can be achieved with Arrow/Polars.
Acceptance criteria:
- DuckDB integration tests use columnar inputs and outputs.
- Pandas is only used where explicitly necessary and justified.

Phase 6 - Test Suite Migration and Cleanup
1) Migrate high-traffic suites first (core, storage, serving).
2) Deprecate tuple-based helpers with clear warnings.
3) Remove unused tuple/pandas helpers once migrations complete.
Acceptance criteria:
- No new tests use tuple-based helpers.
- Legacy helpers are removed or explicitly deprecated.

Phase 7 - Validation and Guardrails
1) Add a contract-alignment check for test-generated readers.
2) Add a linter check or CI guard for JSON stringification in fixtures.
3) Run targeted suites after each phase to ensure parity.
Acceptance criteria:
- Contract alignment errors fail fast with clear messaging.
- No new JSON stringification patterns appear in tests.

Suggested Structural Changes
----------------------------
- Add or extend tests/conftest.py with:
  - schema_service (session)
  - contract_schema_for (session)
  - columnar_rows_for (function)
  - reader_for_rows (function)
- Consolidate helper entry points:
  - tests/_helpers/columnar_streams.py as the primary factory module.
  - tests/_helpers/fixtures/rows.py uses schema-aware normalization only.
- Establish a single assertion module for columnar validation:
  - validate_record_batch_reader and schema metadata checks.

Validation Plan
---------------
Targeted test runs:
- uv run pytest tests/_helpers -q
- uv run pytest tests/core -q
- uv run pytest tests/storage -q
- uv run pytest tests/serving -q

Deliverables
------------
- New/updated conftest fixtures for contract schema and columnar readers.
- Consolidated columnar helper APIs with contract alignment.
- Streaming-safe assertions and IPC-based golden snapshots.
- Removal of tuple-based and pandas-only test pathways.

Open Questions
--------------
1) Default extras policy for tests: retain vs strict.
2) Primary contract source: metadata registry vs TableSchema provider.
3) IPC goldens: store full IPC bytes vs derived JSON summaries for diffing.
