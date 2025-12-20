## ADDED Requirements
### Requirement: Canonical ProblemDetail payload
The system SHALL use the core RFC 9457 ProblemDetail model as the canonical error payload
across CLI, build, serving HTTP, and MCP adapters.

#### Scenario: HTTP uses canonical ProblemDetail
- **WHEN** an HTTP request fails with a domain error
- **THEN** the response payload is derived from the core ProblemDetail model

### Requirement: Catalog-backed error mapping
Serving error codes SHALL map to ProblemDetail type and extension fields (code, kind,
retryable, hint) using the serving error catalog.

#### Scenario: Catalog mapping populates extensions
- **WHEN** a CODEINTEL_SEMANTIC_VIEW_NOT_FOUND error is raised
- **THEN** the ProblemDetail includes the catalog code, kind, and hint extensions

### Requirement: Transport adapters preserve correlation identifiers
Transport adapters SHALL surface correlation/request identifiers in ProblemDetail extensions
without requiring transport-specific error models.

#### Scenario: Correlation ID is preserved
- **WHEN** a request carries a correlation identifier
- **THEN** the ProblemDetail includes that identifier in its extensions
