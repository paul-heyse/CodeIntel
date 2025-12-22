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

