# Codebase Streamlining Inventory

## Pipeline Ownership Map

### Input Normalization (Tabular)
- **Owner**: `src/codeintel/core/columnar/tabular_adapter.py`
- **Responsibilities**: Normalize `TabularInput` to `LazyFrame`, `RecordBatchReader`, or `DuckDBRelation` and handle ephemeral registration.

### Schema Contracts
- **Owner**: `src/codeintel/core/schemas/contracts.py`
- **Responsibilities**: TableSchema ↔ Arrow schema ↔ JSON schema/IPC conversions, contract metadata, hashing.

### Materialization Pipeline
- **Owner**: `src/codeintel/build/hamilton/materializers/base_pipeline.py`
- **Responsibilities**: Shared saver workflow (resolve, validate, write, observe, return results).

### Observation Payloads
- **Owner**: `src/codeintel/storage/tracking/observation_codec.py`
- **Responsibilities**: Typed encode/decode for column stats, dataset stats, derived settings.

### Export Serialization
- **Owner**: `src/codeintel/core/exports/codecs.py` (new boundary)
- **Responsibilities**: Registry-driven encoding for row, batch, and reader outputs.

### Manifest I/O
- **Owner**: `src/codeintel/storage/manifests/manifest_io.py`
- **Responsibilities**: Manifest read/write, path resolution, hash validation.

### Tool Target Scaffolding
- **Owner**: `src/codeintel/build/hamilton/native/patterns/target_builder.py` (new boundary)
- **Responsibilities**: Generate run/ingest/materialize nodes and enforce docstring summary usage for target anchors.

### Config Access
- **Owner**: `src/codeintel/core/config/view.py` (new boundary)
- **Responsibilities**: Typed settings access with defaults and computed values.

## Conversion Call-Site Inventory

### `to_lazyframe` Call Sites
- `src/codeintel/build/schemas/seed_harness.py`
- `src/codeintel/build/hamilton/native/analytics/tables_functions.py`
- `src/codeintel/build/hamilton/native/analytics/tables_modules.py`
- `src/codeintel/build/hamilton/native/analytics/tables_risk.py`
- `src/codeintel/build/hamilton/native/ingestion/frame_utils.py`

### `to_record_batch_reader` Call Sites
- `src/codeintel/storage/warehouse.py`

### `to_relation` Call Sites
- `src/codeintel/core/columnar/tabular_adapter.py` (internal usage for registry)

### `register_ephemeral` Call Sites
- `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`
- `src/codeintel/core/columnar/tabular_adapter.py` (internal usage)

## Notes
- The conversion call-site list focuses on Arrow/Polars/DuckDB normalization helpers.
- Schema alignment helpers (e.g., `align_reader_to_contract`) are tracked under the schema contract boundary.
