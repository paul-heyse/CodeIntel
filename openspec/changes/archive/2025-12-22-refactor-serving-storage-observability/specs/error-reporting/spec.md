## ADDED Requirements
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
