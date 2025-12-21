# config-injection Specification

## Purpose
TBD - created by archiving change refactor-contracts-storage-boundaries. Update Purpose after archive.
## Requirements
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

### Requirement: Canonical configuration identifiers
The system SHALL expose canonical execution profile names and option result types only,
and SHALL NOT provide legacy aliases or compatibility shims.

#### Scenario: Legacy profile alias rejected
- **WHEN** configuration requests the "default" profile alias
- **THEN** profile resolution fails with an unknown-profile error

#### Scenario: ValidationOutcome is the only options result type
- **WHEN** options validation is performed
- **THEN** the result type is ValidationOutcome and no ValidationResult alias is exported

### Requirement: Tool execution dependencies are injected
Tool execution dependencies (ToolService, ToolRunner, and tool configuration) SHALL be
provided via BuildEnv/Providers injection, and modules SHALL NOT instantiate tool runners
or services directly.

#### Scenario: Tool execution uses injected providers
- **WHEN** a module executes an external tool
- **THEN** it uses the injected ToolService/ToolRunner from BuildEnv providers

### Requirement: Analytics resources use injected registry access
Analytics resource loading SHALL be exposed through injected BuildEnv/Providers interfaces
or a unified registry facade, and analytics modules SHALL NOT construct standalone
registries without injection.

#### Scenario: Analytics registry comes from providers
- **WHEN** analytics code requires access to the resource registry
- **THEN** it uses an injected registry or provider facade rather than constructing its own

