# Core Data Format Policy

## Summary
- Internal pipelines use Arrow/Parquet via PyArrow tables, record batches, and readers.
- JSON is boundary-only: DuckDB JSON columns, ingestion payloads, and export artifacts.
- Export serialization is limited to export modules (for example, `codeintel.core.exports`).

## Scope
- Applies to `src/codeintel/core` utilities and all consumers in build, storage, and serving.

## Guidance
- Prefer Arrow schema metadata and Parquet-backed datasets for internal flow.
- Use `codeintel.core.helpers.json` only at boundaries for best-effort JSON parsing.
- Avoid generating or passing JSON objects inside core logic unless required by a boundary.

## Rationale
- Arrow/Parquet provide consistent, typed schemas and faster analytics.
- Boundary-only JSON avoids mixed representations and reduces parsing ambiguity.
