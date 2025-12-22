# Observability

OpenTelemetry is the canonical pipeline for traces and metrics across CLI, HTTP, and MCP.

## Defaults

- DuckDB tracing is enabled by default when OpenTelemetry is enabled.
- DuckDB spans emit `db.query.summary` with SQLGlot-backed summaries and a stable
  `codeintel.db.statement.sha256` hash for grouping.
- DuckDB span names use `db.query.summary` when available.
- DuckDB spans are emitted only when a parent span exists unless explicitly
  configured otherwise.
- Prometheus scraping is opt-in and exposed only when enabled.
- Outbound HTTP client instrumentation is enabled when httpx/requests are installed.

## Environment variables

### Core OpenTelemetry

- `OTEL_SDK_DISABLED` (default: false)
- `OTEL_SERVICE_NAME` (default: per entrypoint, e.g. `codeintel-serving`)
- `OTEL_EXPORTER_OTLP_ENDPOINT` (default: unset)
- `CODEINTEL_EXPORT_TRACES` (default: true)
- `CODEINTEL_EXPORT_METRICS` (default: true)
- `CODEINTEL_CONSOLE_TELEMETRY` (default: false)

### Prometheus export

- `CODEINTEL_PROMETHEUS_METRICS` (default: false)
  - When enabled and `prometheus_client` is installed, `/metrics` is exposed for HTTP and MCP.
- `CODEINTEL_METRICS_REQUIRE_AUTH` (default: false)
  - When true, the HTTP `/metrics` route enforces the configured auth token or API key.

### DuckDB tracing

- `CODEINTEL_OTEL_DUCKDB_TRACING` (default: true)
  - When true and OpenTelemetry is enabled, DuckDB spans are emitted.
- `CODEINTEL_OTEL_DUCKDB_REQUIRE_PARENT` (default: true)
  - When true, DuckDB spans are emitted only when a parent span exists.
- `CODEINTEL_OTEL_DB_STATEMENT_MODE` (default: `hash`)
  - Options: `full`, `hash`, `operation`, `none`.
- `CODEINTEL_OTEL_DB_STATEMENT_HASH_LEN` (default: 16)
  - Controls the display prefix length when statement mode is `hash`.
- `CODEINTEL_OTEL_DB_QUERY_SUMMARY_MAX_LEN` (default: 255)
  - Token-safe truncation limit for `db.query.summary`.
- `CODEINTEL_OTEL_DB_QUERY_SUMMARY_MAX_TARGETS` (default: 6)
  - Maximum number of table targets included per operation.
- `CODEINTEL_OTEL_DB_QUERY_SUMMARY_EMIT_ELLIPSIS` (default: true)
  - Append `...` when the summary truncates or target cap is hit.
- `CODEINTEL_OTEL_DB_QUERY_SUMMARY_HASH_SUSPICIOUS` (default: true)
  - Hash suspicious/high-cardinality targets (paths/URLs/long identifiers).
- `CODEINTEL_OTEL_DB_QUERY_SUMMARY_HASH_LEN` (default: 12)
  - Length of hashed target identifiers in summaries.
- `CODEINTEL_OTEL_DB_QUERY_SUMMARY_HASH_MIN_LEN` (default: 64)
  - Minimum identifier length that triggers hashing (when enabled).
- `CODEINTEL_OTEL_DB_QUERY_SUMMARY_INCLUDE_SUBQUERY_OPS` (default: true)
  - Include nested subquery operations (e.g., `SELECT SELECT orders`).
- `CODEINTEL_OTEL_DB_QUERY_SUMMARY_INCLUDE_MULTI_STATEMENT` (default: true)
  - Include multiple statements (semicolon-separated) in summaries.
- `CODEINTEL_OTEL_DB_QUERY_SUMMARY_SPAN_NAME_HOOK` (default: false)
  - Update DB span names to `db.query.summary` for DBAPI instrumentation.
- `CODEINTEL_OTEL_DB_LEGACY_ATTRIBUTES` (default: false)
  - When true, emit legacy `db.system`/`db.name` attributes in addition to new keys.
- `CODEINTEL_OTEL_DB_QUERY_TEXT_POLICY` (default: `never`)
  - Options: `never`, `parameterized`, `redacted`, `parameterized_or_redacted`, `full`.
- `CODEINTEL_OTEL_DB_QUERY_TEXT_MAX_LEN` (default: 4096)
  - Length cap for sanitized `db.query.text`.
- `CODEINTEL_OTEL_DB_QUERY_TEXT_STRIP_COMMENTS` (default: true)
  - Remove SQL comments before sanitization.
- `CODEINTEL_OTEL_DB_QUERY_TEXT_COLLAPSE_IN_LISTS` (default: true)
  - Collapse repeated `IN (?, ?, ?)` placeholder lists to `IN (?)`.
- `CODEINTEL_OTEL_DB_QUERY_PARAMETER_ENABLED` (default: false)
  - When true, emit `db.query.parameter.<key>` attributes for allowlisted keys.
- `CODEINTEL_OTEL_DB_QUERY_PARAMETER_KEYS` (default: empty)
  - Comma-separated allowlist of parameter keys to emit.
- `CODEINTEL_OTEL_DB_QUERY_PARAMETER_HASH_KEYS` (default: empty)
  - Comma-separated keys whose string values should be hashed.
- `CODEINTEL_OTEL_DB_QUERY_PARAMETER_REQUIRE_IN_SQL` (default: true)
  - When true, only emit allowlisted keys that appear as placeholders in the SQL.
- `CODEINTEL_OTEL_DB_QUERY_PARAMETER_MAX_STRLEN` (default: 80)
  - Maximum length for emitted string parameter values.

## Operational guidance

- Prefer dashboards keyed by `db.query.summary` plus `codeintel.db.statement.sha256`
  for stable grouping without raw SQL text.
