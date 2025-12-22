## ADDED Requirements
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

## MODIFIED Requirements
### Requirement: Canonical runtime configuration loader
The system SHALL provide a single runtime configuration loader that returns RuntimePrimitives
and settings for build, serving, and CLI entrypoints, and environment parsing SHALL be
confined to that loader plus the explicit CLI config env allowlist used for top-level
overrides.

#### Scenario: Runtime loader centralizes environment parsing
- **WHEN** a CLI entrypoint constructs runtime primitives or CLI config
- **THEN** it uses the canonical loader and the only permitted env parsing outside it is
  the top-level CLI config override allowlist
