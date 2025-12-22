## ADDED Requirements
### Requirement: Semantic query responses include SQL fingerprints
Serving semantic query responses SHALL include sql_fingerprint computed from canonicalized
SQL when compiled SQL is available, and SHALL omit sql_fingerprint when compilation fails.

#### Scenario: SQL fingerprint is emitted when SQL is available
- **WHEN** a semantic query compiles SQL successfully
- **THEN** the response includes sql_fingerprint from the canonical fingerprint pipeline

#### Scenario: SQL fingerprint is omitted when SQL is unavailable
- **WHEN** SQL compilation fails for a semantic query
- **THEN** the response omits sql_fingerprint

### Requirement: MCP export reads return ResourceContent with metadata
MCP export resource reads SHALL return ResourceContent with the export MIME type from the
canonical format registry and include export metadata (export_id, row_count, size_bytes).
Binary exports MUST return bytes and text exports MUST return UTF-8 text.

#### Scenario: Binary export read returns explicit MIME and metadata
- **WHEN** a parquet export resource is read
- **THEN** the response is ResourceContent with bytes, the parquet MIME type, and metadata

### Requirement: Correlation IDs are generated when missing
Serving transports SHALL generate a correlation ID via the canonical ID factory when one is
not provided and SHALL surface it in HTTP response headers and MCP error contexts.

#### Scenario: HTTP response includes generated correlation ID
- **WHEN** an HTTP request arrives without X-Correlation-ID
- **THEN** the response includes a generated X-Correlation-ID header

#### Scenario: MCP errors include generated correlation ID
- **WHEN** an MCP call fails without a prior correlation identifier
- **THEN** the error context includes a generated correlation ID
