## ADDED Requirements
### Requirement: DB span emission is centralized and extensible
Database spans SHALL be emitted through a shared DB span emitter that encapsulates
attribute composition, policy decisions, and redaction rules. DuckDB tracing SHALL
use this emitter and future database adapters SHALL integrate through the same
interface.

#### Scenario: DuckDB uses the shared span emitter
- **WHEN** a DuckDB query span is created
- **THEN** the span attributes and redaction behavior come from the shared emitter
