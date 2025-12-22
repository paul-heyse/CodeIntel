## ADDED Requirements
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
