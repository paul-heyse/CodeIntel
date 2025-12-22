## ADDED Requirements
### Requirement: MCP semantic tools accept request envelopes
The MCP semantic tools SHALL accept a single request object that conforms to the
semantic request models and SHALL validate inputs before execution.

#### Scenario: Semantic query validates request envelope
- **WHEN** a client calls semantic_query with a request payload
- **THEN** the payload is validated as SemanticQueryRequest and invalid fields are rejected

#### Scenario: Semantic explain validates request envelope
- **WHEN** a client calls semantic_explain with a request payload
- **THEN** the payload is validated as SemanticQueryRequest and invalid fields are rejected

#### Scenario: Semantic export validates request envelope
- **WHEN** a client calls semantic_export with a request payload
- **THEN** the payload is validated as SemanticExportRequest and invalid fields are rejected

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
