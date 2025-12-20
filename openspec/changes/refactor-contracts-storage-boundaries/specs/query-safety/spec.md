## ADDED Requirements
### Requirement: Safe query helpers never raise on invalid input
Safe query helpers SHALL return None/False for invalid table keys and SHALL NOT raise
exceptions for invalid input or SQL-injection probes.

#### Scenario: Safe count with invalid key
- **WHEN** safe_count receives an invalid or malicious table key
- **THEN** it returns None and no exception is raised

#### Scenario: Safe exists with invalid key
- **WHEN** safe_table_exists receives an invalid or malicious table key
- **THEN** it returns False and no exception is raised

### Requirement: Explicit table-key validation API
The system SHALL provide strict table-key parsing that raises a typed validation error and
safe validation helpers that return a boolean or result object.

#### Scenario: Strict validation rejects invalid keys
- **WHEN** a strict table-key parser receives an invalid key
- **THEN** it raises a typed validation error

#### Scenario: Safe validation returns False
- **WHEN** a safe table-key validator receives an invalid key
- **THEN** it returns False without raising

### Requirement: Queries execute only for validated keys
The system SHALL execute SQL only after a table key has been validated.

#### Scenario: Query path rejects invalid keys
- **WHEN** a table key fails validation
- **THEN** no SQL query is executed for that key
