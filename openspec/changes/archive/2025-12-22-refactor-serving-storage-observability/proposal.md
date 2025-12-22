# Change: Refactor serving/storage observability and SQL tooling

## Why
Serving, storage, and CLI observability are currently fragmented, and SQL fingerprinting
logic is duplicated across layers. Consolidating these cross-cutting concerns improves
operational clarity, removes drift in query fingerprints, and makes export and error
surfaces more consistent.

## What Changes
- Centralize observability bootstrap for CLI, HTTP, and MCP with OpenTelemetry as the
  canonical metrics pipeline.
- Enable DuckDB DB-API tracing by default when OTel is enabled, with safe SQL redaction
  and configurable statement modes.
- Unify SQL fingerprinting through storage SQLGlot tools with safe fallbacks and add
  semantic SQL diffs for upgrade diagnostics.
- Standardize ID generation with a canonical UUID factory (UUIDv7 when available) across
  correlation IDs, error instance/debug IDs, and run/job IDs.
- Improve export responses: MCP reads return content with registry MIME, export metadata is
  served via export meta resources/resource listing _meta, and NDJSON uses a shared UTF-8
  encoder with a msgspec fast path.

## Impact
- Affected specs: observability (new), serving-interfaces, export-formats,
  storage-boundaries, error-reporting
- Affected code: src/codeintel/observability/*, src/codeintel/cli/observability/*,
  src/codeintel/serving/http/*, src/codeintel/serving/mcp/*, src/codeintel/serving/metrics.py,
  src/codeintel/serving/semantic/fingerprints.py, src/codeintel/storage/sqlglot_tools.py,
  src/codeintel/storage/backend/duckdb_session.py, src/codeintel/core/execution/ids.py,
  tests/storage/test_sql_compiler_upgrade_gates.py, tests/observability/*,
  tests/serving/test_streaming_ndjson.py
