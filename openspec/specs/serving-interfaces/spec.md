# serving-interfaces Specification

## Purpose
TBD - created by archiving change remove-legacy-compat-code. Update Purpose after archive.
## Requirements
### Requirement: Direct FastMCP imports
Serving MCP components SHALL import FastMCP types directly from fastmcp packages and
SHALL NOT rely on local compatibility shims.

#### Scenario: MCP server uses direct imports
- **WHEN** the MCP server is constructed
- **THEN** FastMCP, Context, and EventStore are imported from fastmcp directly

### Requirement: MCP semantic tools accept request envelopes
The MCP semantic tools SHALL accept a request envelope containing a request payload,
validate it with MCP-specific request models, and normalize it into the semantic request
models before execution.

#### Scenario: Semantic query validates request envelope
- **WHEN** a client calls semantic_query with a request payload
- **THEN** the payload is validated as SemanticQueryToolRequest and normalized to
  SemanticQueryRequest, rejecting invalid fields

#### Scenario: Semantic explain validates request envelope
- **WHEN** a client calls semantic_explain with a request payload
- **THEN** the payload is validated as SemanticQueryToolRequest and normalized to
  SemanticQueryRequest, rejecting invalid fields

#### Scenario: Semantic export validates request envelope
- **WHEN** a client calls semantic_export with a request payload
- **THEN** the payload is validated as SemanticExportToolRequest and normalized to
  SemanticExportRequest, rejecting invalid fields

### Requirement: Health and readiness routes are async-safe
MCP health and readiness routes SHALL await readiness signaling and SHALL respond
using cached snapshot metadata without blocking the event loop.

#### Scenario: Health returns cached snapshot metadata
- **WHEN** the server is ready and /health is requested
- **THEN** the response includes cached snapshot metadata for repo, commit, and run_id

#### Scenario: Ready returns unavailable when snapshot is missing
- **WHEN** no snapshot is available and /ready is requested
- **THEN** the response is 503 with a not-ready status

### Requirement: Prompt registry introspection API
Serving MCP prompt registration SHALL expose a public API to enumerate registered
prompt names without accessing private FastMCP state.

#### Scenario: Prompt registry lists registered prompts
- **WHEN** prompts are registered with the MCP server
- **THEN** the public prompt registry API returns their names

#### Scenario: Prompt registry is empty with no prompts
- **WHEN** no prompts are registered
- **THEN** the public prompt registry API returns an empty set

### Requirement: Semantic query responses include SQL fingerprints
Serving semantic query responses SHALL include sql_fingerprint computed from canonicalized
SQL when compiled SQL is available, and SHALL omit sql_fingerprint when compilation fails.

#### Scenario: SQL fingerprint is emitted when SQL is available
- **WHEN** a semantic query compiles SQL successfully
- **THEN** the response includes sql_fingerprint from the canonical fingerprint pipeline

#### Scenario: SQL fingerprint is omitted when SQL is unavailable
- **WHEN** SQL compilation fails for a semantic query
- **THEN** the response omits sql_fingerprint

### Requirement: MCP export reads return content with registry MIME; metadata is available
MCP export resource reads SHALL return content with the export MIME type from the
canonical format registry. Export metadata (export_id, row_count, size_bytes) SHALL be
available via the export meta resource and MAY be surfaced in resource listing _meta.
Binary exports MUST return bytes and text exports MUST return UTF-8 text.

#### Scenario: Binary export read returns explicit MIME
- **WHEN** a parquet export resource is read
- **THEN** the response returns bytes with the parquet MIME type

#### Scenario: Export metadata is available via meta resource
- **WHEN** a client reads codeintel://exports/{export_id}/meta
- **THEN** the response includes export_id, row_count, and size_bytes metadata

### Requirement: Correlation IDs are generated when missing
Serving transports SHALL generate a correlation ID via the canonical ID factory when one is
not provided and SHALL surface it in HTTP response headers and MCP error contexts.

#### Scenario: HTTP response includes generated correlation ID
- **WHEN** an HTTP request arrives without X-Correlation-ID
- **THEN** the response includes a generated X-Correlation-ID header

#### Scenario: MCP errors include generated correlation ID
- **WHEN** an MCP call fails without a prior correlation identifier
- **THEN** the error context includes a generated correlation ID

### Requirement: Serving uses the canonical registry service
Serving SHALL source semantic catalogs, export metadata, and schema summaries from the
canonical RegistryService and shared manifests. Serving SHALL NOT compile or maintain
local registry copies.

#### Scenario: Serving catalog derives from registry service
- **WHEN** a serving catalog response is generated
- **THEN** it uses RegistryService outputs backed by the canonical metadata tables

