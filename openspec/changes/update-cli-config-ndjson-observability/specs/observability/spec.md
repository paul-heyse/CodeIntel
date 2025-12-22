## ADDED Requirements
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
