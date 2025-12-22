## ADDED Requirements
### Requirement: Centralized observability bootstrap
The system SHALL provide a single observability bootstrap used by CLI, HTTP, and MCP
entrypoints to configure OpenTelemetry tracing and metrics, and SHALL degrade safely to a
no-op when OpenTelemetry is unavailable or disabled.

#### Scenario: CLI, HTTP, and MCP share bootstrap
- **WHEN** a CLI command, HTTP server, or MCP server starts
- **THEN** the same bootstrap initializes tracing and metrics once per process or no-ops
  when disabled

### Requirement: OpenTelemetry metrics are canonical
The system SHALL treat OpenTelemetry metrics as the canonical metrics pipeline and SHALL
record query and operation metrics via shared helpers using low-cardinality labels.

#### Scenario: Query metrics emit OTel counters and histograms
- **WHEN** log_query_metrics is invoked for a semantic query
- **THEN** OTel metrics record endpoint and component labels without query text or hashes

### Requirement: Prometheus scrape endpoints are OTel-backed
When the OpenTelemetry Prometheus exporter is enabled, serving HTTP and MCP SHALL expose a
/metrics endpoint returning OTel-collected metrics; when disabled, /metrics SHALL NOT be
exposed.

#### Scenario: Metrics endpoint is present when enabled
- **WHEN** the Prometheus exporter is enabled
- **THEN** /metrics returns OTel metrics for serving HTTP and MCP

#### Scenario: Metrics endpoint is absent when disabled
- **WHEN** the Prometheus exporter is disabled
- **THEN** /metrics is not registered

### Requirement: DuckDB tracing defaults to redacted spans
DuckDB DB-API tracing SHALL be enabled by default when OpenTelemetry is enabled and SHALL
redact db.statement by emitting an operation + hash display and a stable SHA-256 attribute.
Tracing MAY be disabled via environment toggles.

#### Scenario: Default tracing redacts SQL statements
- **WHEN** a DuckDB query executes with OpenTelemetry enabled and no overrides
- **THEN** db.statement contains an operation + hash and codeintel.db.statement.sha256 is set

#### Scenario: Tracing can be disabled
- **WHEN** CODEINTEL_OTEL_DUCKDB_TRACING is set to false
- **THEN** DuckDB connections are not instrumented

### Requirement: Correlation context is propagated
The system SHALL propagate correlation IDs through a shared context and SHALL attach them
to spans or baggage when available across CLI, HTTP, and MCP.

#### Scenario: HTTP correlation ID reaches spans
- **WHEN** an HTTP request includes X-Correlation-ID
- **THEN** the correlation ID is available in context and attached to the active span

### Requirement: Outbound HTTP instrumentation is best-effort
The observability bootstrap SHALL attempt to instrument outbound HTTP clients (httpx and
requests) when present and SHALL no-op when those packages are unavailable.

#### Scenario: HTTP client instrumentation is enabled when available
- **WHEN** the bootstrap runs and httpx is installed
- **THEN** outbound httpx requests are instrumented without additional caller changes
