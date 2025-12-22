# observability Specification

## Purpose
TBD - created by archiving change refactor-serving-storage-observability. Update Purpose after archive.
## Requirements
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

### Requirement: DuckDB spans emit query summaries for grouping
DuckDB spans SHALL include a `db.query.summary` attribute generated from SQLGlot parsing
with alias normalization, CTE-safe table extraction, multi-operation summaries, and
token-safe truncation at 255 characters. The span name SHALL be set to
`db.query.summary` when present.

#### Scenario: Alias-normalized summary
- **WHEN** a DuckDB query uses table aliases (explicit or implicit)
- **THEN** `db.query.summary` includes the base table names without alias tokens
  and the span name matches the summary

#### Scenario: Multi-operation summary
- **WHEN** a query performs INSERT ... SELECT
- **THEN** `db.query.summary` contains both operations and their targets in-order
  and the span name matches the summary

#### Scenario: Token-safe truncation
- **WHEN** a summary would exceed 255 characters
- **THEN** the emitted summary is truncated at a token boundary and is 255
  characters or fewer

### Requirement: Centralized DB span attribute composition
DuckDB spans SHALL be composed through a shared attribute builder that sets
`db.system.name`, `db.namespace`, `db.query.summary`, and
`codeintel.db.statement.sha256`. The builder SHALL NOT derive `db.operation.name`
from SQL text. Legacy `db.system`/`db.name` attributes MAY be emitted only when
explicitly enabled.

#### Scenario: Builder emits canonical attributes
- **WHEN** a DuckDB query span is created
- **THEN** the span includes `db.system.name`, `db.namespace`, `db.query.summary`,
  and `codeintel.db.statement.sha256`
- **AND** `db.operation.name` is absent unless provided by a higher-level caller

### Requirement: Shared SQL canonicalization for summaries and hashes
SQL summaries and statement hashes SHALL use a single SQLGlot canonicalization
pipeline, and SHALL fall back to safe normalization and raw SHA-256 hashing when
parsing fails.

#### Scenario: Parse failure still yields a hash
- **WHEN** SQL parsing fails for a statement
- **THEN** `codeintel.db.statement.sha256` is still emitted via the fallback path

### Requirement: Opt-in sanitized db.query.text emission
The system SHALL NOT emit `db.query.text` by default. When enabled, `db.query.text`
SHALL be emitted only for parameterized queries or SQLGlot-redacted queries with
literal placeholders and a length cap. Raw literal SQL SHALL NOT be emitted unless
explicitly configured for debug-only use.

#### Scenario: Default suppresses query text
- **WHEN** DuckDB spans are emitted with default settings
- **THEN** `db.query.text` is absent

#### Scenario: Parameterized policy emits query text
- **WHEN** `db.query.text` emission is set to parameterized-only and the SQL uses
  placeholders with parameters provided
- **THEN** `db.query.text` is emitted with the parameterized SQL text

#### Scenario: Redacted policy emits sanitized text
- **WHEN** `db.query.text` emission is set to redacted and SQL includes literals
- **THEN** `db.query.text` is emitted with literals replaced by placeholders

### Requirement: Opt-in allowlisted db.query.parameter emission
The system SHALL emit `db.query.parameter.<key>` attributes only when explicitly
enabled, only for named parameters, only for allowlisted keys, and only for scalar
values with length limits. Batch executions SHALL NOT emit parameter attributes.

#### Scenario: Allowlisted named parameters only
- **WHEN** a query executes with named parameters and an allowlist is configured
- **THEN** only allowlisted keys are emitted as `db.query.parameter.<key>`

#### Scenario: Batch execution emits no parameters
- **WHEN** a batch execution occurs (executemany)
- **THEN** no `db.query.parameter.<key>` attributes are emitted

### Requirement: DuckDB spans record errors and correlation IDs
DuckDB spans SHALL record exceptions and set error status on query failures, and
SHALL attach `codeintel.correlation_id` when available in context.

#### Scenario: Failed query marks span error
- **WHEN** a DuckDB query raises an exception
- **THEN** the span records the exception and is marked with error status

#### Scenario: Correlation ID is attached
- **WHEN** a correlation ID is present in context
- **THEN** the DuckDB span includes `codeintel.correlation_id`

### Requirement: Parent-span gating for DuckDB spans
DuckDB spans SHALL be emitted only when an active parent span exists by default.
Configuration SHALL allow always-on emission when required.

#### Scenario: No parent span suppresses DB spans
- **WHEN** no parent span is active and parent-gating is enabled
- **THEN** DuckDB spans are not emitted

#### Scenario: Always-on override emits spans
- **WHEN** parent-gating is disabled by configuration
- **THEN** DuckDB spans are emitted even without a parent span

### Requirement: DB telemetry configuration is exposed
Observability settings SHALL expose configuration for query summary limits,
`db.query.text` policies, `db.query.parameter` allowlists, legacy attribute
emission, and parent-span gating via environment variables and runtime settings.

#### Scenario: Environment toggles override defaults
- **WHEN** DB telemetry environment variables are set
- **THEN** DuckDB telemetry behavior follows the configured policies

### Requirement: DuckDB tracing honors parent span and statement mode toggles
DuckDB tracing SHALL emit spans when tracing is enabled and duckdb_require_parent_span is
false, even if no parent span exists. The statement_mode setting SHALL control db.statement
redaction, and db.query.summary SHALL match the span name.

#### Scenario: Parent span not required
- **WHEN** duckdb_require_parent_span is false and a DuckDB query runs without an active span
- **THEN** a DuckDB span is emitted with db.system.name set to "duckdb"

#### Scenario: Operation statement mode emits SQL operation
- **WHEN** duckdb_statement_mode is set to "operation"
- **THEN** db.statement equals the SQL operation (for example, SELECT) and db.query.summary
  equals the span name

### Requirement: HTTP and MCP spans include correlation IDs
HTTP route wrappers and MCP middleware SHALL emit spans with stable operation names and
attach codeintel.correlation_id when available.

#### Scenario: HTTP span includes correlation ID
- **WHEN** an HTTP request with a correlation ID is handled
- **THEN** the span name is "http.<route>" and codeintel.correlation_id is attached

#### Scenario: MCP span includes correlation ID
- **WHEN** an MCP tool call is handled
- **THEN** the span name is "mcp.tools/call:<tool>" and codeintel.correlation_id is attached

### Requirement: Function effects computation logs population summary
Function effects computation SHALL emit an INFO log record that includes the phrase
"function_effects populated" and the number of rows assembled for the snapshot.

#### Scenario: Function effects info log emitted
- **WHEN** function effects rows are built for a snapshot
- **THEN** an INFO log record includes "function_effects populated"

