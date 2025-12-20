## ADDED Requirements
### Requirement: Explicit settings injection
Runtime settings (e.g., engine version, contract resolution options) SHALL be provided
through explicit settings objects rather than environment lookups at import time.

#### Scenario: Settings control engine version
- **WHEN** a settings object specifies an engine version
- **THEN** the build pipeline uses that version without reading environment variables

### Requirement: Import-time safety
Module imports SHALL NOT perform heavy initialization, DAG construction, or external I/O.

#### Scenario: Import does not initialize the DAG
- **WHEN** build and schema modules are imported
- **THEN** the Hamilton DAG is not constructed during import

### Requirement: Dependency injection for tests
Public APIs SHALL allow injection of contract and metadata providers to enable testing
without monkeypatching.

#### Scenario: Tests use injected providers
- **WHEN** a test passes an injected contract provider
- **THEN** behavior is controlled without monkeypatching
