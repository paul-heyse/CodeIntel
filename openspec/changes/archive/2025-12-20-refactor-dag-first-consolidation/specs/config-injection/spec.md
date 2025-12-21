## MODIFIED Requirements
### Requirement: Explicit settings injection
Runtime settings SHALL be provided through explicit settings objects injected at boundary
entrypoints (BuildEnv/ServingRuntime/ConfigRegistry) rather than implicit environment lookups
inside library modules.

#### Scenario: Settings control engine version
- **WHEN** a settings object specifies an engine version
- **THEN** the build pipeline uses that version without reading environment variables

### Requirement: Import-time safety
Module imports SHALL NOT perform heavy initialization, DAG construction, external I/O, or
settings resolution from environment variables.

#### Scenario: Import does not initialize the DAG
- **WHEN** build and schema modules are imported
- **THEN** the Hamilton DAG is not constructed during import

### Requirement: Dependency injection for tests
Public APIs SHALL allow injection of contract providers, metadata providers, and settings
objects to enable testing without monkeypatching.

#### Scenario: Tests use injected providers
- **WHEN** a test passes injected providers and settings
- **THEN** behavior is controlled without monkeypatching
