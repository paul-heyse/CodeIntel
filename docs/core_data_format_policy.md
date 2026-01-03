# Core Data Format Policy

## Summary
- Internal pipelines use Arrow/Parquet via PyArrow tables, record batches, and readers.
- JSON is boundary-only: DuckDB JSON columns, ingestion payloads, and export artifacts.
- Export serialization is limited to export modules (for example, `codeintel.core.exports`).
- Core does not import `json`/`orjson` outside boundary helpers.
- Avoid materializing Arrow arrays with `to_pylist()` in core logic.

## Scope
- Applies to `src/codeintel/core` utilities and all consumers in build, storage, and serving.

## Guidance
- Prefer Arrow schema metadata and Parquet-backed datasets for internal flow.
- Use `codeintel.core.helpers.json` only at boundaries for best-effort JSON parsing.
- Avoid generating or passing JSON objects inside core logic unless required by a boundary.
- Control schema promotion behavior via `CODEINTEL_BUILD_SCHEMA_PROMOTE_OPTIONS` and
  `CODEINTEL_SERVE_DATASET_SCHEMA_PROMOTE_OPTIONS` (default: `permissive`).

## Rationale
- Arrow/Parquet provide consistent, typed schemas and faster analytics.
- Boundary-only JSON avoids mixed representations and reduces parsing ambiguity.
