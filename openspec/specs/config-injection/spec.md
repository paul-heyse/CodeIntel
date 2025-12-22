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
Analytics and graph resource loading SHALL use a single registry interface supplied by
BuildEnv/Providers, and modules SHALL NOT construct standalone registry implementations. Build
SHALL use the core ResourceRegistry and shared ProviderFactory interface without a build-only
wrapper.

#### Scenario: Analytics registry comes from providers
- **WHEN** analytics or graph code requires access to the resource registry
- **THEN** it uses the injected registry interface from providers rather than constructing one

### Requirement: Canonical runtime configuration loader
The system SHALL provide a single runtime configuration loader that returns RuntimePrimitives
and settings for build, serving, and CLI entrypoints, and environment parsing SHALL be
confined to that loader plus the explicit CLI config env allowlist used for top-level
overrides.

#### Scenario: Runtime loader centralizes environment parsing
- **WHEN** a CLI entrypoint constructs runtime primitives or CLI config
- **THEN** it uses the canonical loader and the only permitted env parsing outside it is
  the top-level CLI config override allowlist

### Requirement: CLI config env overrides are top-level and explicit
The system SHALL apply CLI configuration overrides from an explicit allowlist of top-level
CODEINTEL_* environment variables and SHALL NOT infer nested section overrides from the
environment. Unrecognized keys MUST be ignored.

#### Scenario: Env overrides file config
- **WHEN** CODEINTEL_COLOR is set to "false" and a config file sets color: true
- **THEN** the resolved config uses color=false and records an env source

#### Scenario: Unknown env keys are ignored
- **WHEN** CODEINTEL_PROGRESS_ENABLED is set but not allowlisted
- **THEN** the resolved config retains file/default values for progress.enabled

### Requirement: Cyclopts config chain mirrors CLI precedence
The Cyclopts App config chain SHALL include the optional TOML loader and the env override
loader so that CLI parsing mirrors config precedence.

#### Scenario: Config chain includes two loaders
- **WHEN** the CLI root app is constructed
- **THEN** app.config includes two entries ordered for TOML first and env overrides second

