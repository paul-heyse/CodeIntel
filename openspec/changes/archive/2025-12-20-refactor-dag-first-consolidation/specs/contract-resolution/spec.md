## MODIFIED Requirements
### Requirement: Schema-only contract enumeration is DAG-free
Schema-only contract enumeration SHALL NOT initialize the Hamilton DAG or load target
metadata. Output inventory and exclusion lists SHALL be derived from target specs or injected
providers that do not require DAG construction.

#### Scenario: Schema-only enumeration avoids DAG initialization
- **WHEN** schema-only contracts are enumerated or default validation schemas are requested
- **THEN** the Hamilton DAG is not constructed

### Requirement: Lazy metadata enrichment
Metadata enrichment SHALL be lazy and only initialize the Hamilton DAG when explicitly
requested via injected metadata providers.

#### Scenario: Metadata requested triggers DAG initialization
- **WHEN** metadata enrichment is requested for a contract
- **THEN** the Hamilton DAG initializes to provide the metadata

### Requirement: Injectable metadata providers
The system SHALL allow dependency injection of metadata and output inventory providers to
support testability and alternative runtime implementations.

#### Scenario: Tests inject a stub metadata provider
- **WHEN** a test supplies a stub metadata provider and output inventory
- **THEN** contract enumeration succeeds without DAG initialization
