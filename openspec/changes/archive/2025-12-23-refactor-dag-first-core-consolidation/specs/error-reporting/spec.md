## ADDED Requirements
### Requirement: Error catalog is canonical across transports
CLI, build, HTTP, and MCP error handling SHALL use a single canonical error catalog
and mapping pipeline. Layer-specific catalogs or duplicated error enums SHALL NOT be
introduced.

#### Scenario: CLI and HTTP share catalog mappings
- **WHEN** the same domain error is raised via CLI and HTTP
- **THEN** both surfaces map it through the canonical catalog with identical
  ProblemDetail codes and extensions
