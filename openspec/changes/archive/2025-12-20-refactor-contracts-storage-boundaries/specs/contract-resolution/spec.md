## ADDED Requirements
### Requirement: Schema-only contract enumeration is DAG-free
Schema-only contract enumeration SHALL NOT initialize the Hamilton DAG or load target
metadata.

#### Scenario: Schema-only enumeration avoids DAG initialization
- **WHEN** schema-only contracts are enumerated
- **THEN** the Hamilton DAG is not constructed

### Requirement: Lazy metadata enrichment
Metadata enrichment SHALL be lazy and only initialize the Hamilton DAG when explicitly
requested.

#### Scenario: Metadata requested triggers DAG initialization
- **WHEN** metadata enrichment is requested for a contract
- **THEN** the Hamilton DAG initializes to provide the metadata

### Requirement: Injectable metadata providers
The system SHALL allow dependency injection of metadata providers to support testability
and alternative runtime implementations.

#### Scenario: Tests inject a stub metadata provider
- **WHEN** a test supplies a stub metadata provider
- **THEN** contract enumeration succeeds without DAG initialization
