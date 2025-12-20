## ADDED Requirements

### Requirement: ProblemDetail is the canonical error envelope
All build, HTTP, and MCP error responses SHALL be represented as RFC 9457 ProblemDetail payloads
with type, title, status, detail, instance, and extensions fields.

#### Scenario: HTTP error payload
- **WHEN** a serving request fails
- **THEN** the response body is a ProblemDetail JSON object

#### Scenario: Build error payload
- **WHEN** a build operation fails with a contract error
- **THEN** the error is represented as a ProblemDetail payload with an error code

### Requirement: Single error catalog
The system SHALL maintain a single error catalog defining stable codes, titles, and default
statuses for ProblemDetail creation.

#### Scenario: Error code mapping
- **WHEN** an error code is requested
- **THEN** the catalog provides the canonical title and status for that code

### Requirement: Correlation and debug identifiers
ProblemDetail instances SHALL include correlation or instance identifiers to support tracing.

#### Scenario: Instance identifier
- **WHEN** a ProblemDetail is created without an explicit instance
- **THEN** the system generates a unique instance identifier
