# Observability

OpenTelemetry is the canonical pipeline for traces and metrics across CLI, HTTP, and MCP.

## Defaults

- DuckDB tracing is enabled by default when OpenTelemetry is enabled.
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
- `CODEINTEL_OTEL_DB_STATEMENT_MODE` (default: `hash`)
  - Options: `full`, `hash`, `operation`, `none`.
- `CODEINTEL_OTEL_DB_STATEMENT_HASH_LEN` (default: 16)
  - Controls the display prefix length when statement mode is `hash`.
