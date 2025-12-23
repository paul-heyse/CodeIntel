# error-reporting Specification

## Purpose
TBD - created by archiving change refactor-dag-first-consolidation. Update Purpose after archive.
## Requirements
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

### Requirement: Error taxonomy is imported from core
CLI error handling SHALL import taxonomy definitions from the core taxonomy module and SHALL
NOT re-export or shadow taxonomy types in CLI-specific modules.

#### Scenario: CLI taxonomy uses core module
- **WHEN** CLI error handlers need taxonomy definitions
- **THEN** they import from the core taxonomy module

### Requirement: Build and serving errors share core taxonomy
Build and serving error types SHALL map to the core ProblemDetail taxonomy and shared error
codes, and layer-specific error catalogs SHALL NOT redefine or shadow core error definitions.

#### Scenario: Build errors use core taxonomy
- **WHEN** a build error is raised and rendered for CLI or API output
- **THEN** the ProblemDetail payload uses core taxonomy codes and extensions

### Requirement: ProblemDetail identifiers use canonical ID factory
ProblemDetail instance identifiers and transport debug IDs SHALL be generated via the
canonical ID factory (preferring UUIDv7 when available) and included when absent from the
error context.

#### Scenario: Debug ID is generated when missing
- **WHEN** an error is mapped without a debug ID
- **THEN** a new canonical debug ID is generated and included in ProblemDetail extensions

#### Scenario: Instance ID uses canonical factory
- **WHEN** a ProblemDetail instance ID is generated
- **THEN** it uses the canonical ID factory and yields a stable UUID string

### Requirement: Error catalog is canonical across transports
CLI, build, HTTP, and MCP error handling SHALL use a single canonical error catalog
and mapping pipeline. Layer-specific catalogs or duplicated error enums SHALL NOT be
introduced.

#### Scenario: CLI and HTTP share catalog mappings
- **WHEN** the same domain error is raised via CLI and HTTP
- **THEN** both surfaces map it through the canonical catalog with identical
  ProblemDetail codes and extensions

