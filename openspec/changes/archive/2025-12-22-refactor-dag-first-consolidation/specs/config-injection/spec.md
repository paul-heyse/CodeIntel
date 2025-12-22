## MODIFIED Requirements
### Requirement: Canonical runtime configuration loader
The system SHALL provide a single runtime configuration loader that returns RuntimePrimitives
and settings for build, serving, and CLI entrypoints, and environment parsing SHALL be
confined to that loader. Observability and metrics gating settings SHALL be sourced from the
same loader, and entry points SHALL NOT perform bespoke env/path parsing outside the
canonical loader.

#### Scenario: Runtime loader centralizes CLI parsing
- **WHEN** a CLI entrypoint constructs runtime primitives
- **THEN** it uses the canonical loader and no per-surface env parsing modules are used

#### Scenario: Runtime loader centralizes serving parsing
- **WHEN** a serving entrypoint constructs runtime primitives
- **THEN** it uses the canonical loader and no library modules parse environment variables

#### Scenario: Observability settings are loader-driven
- **WHEN** observability bootstrap is invoked for CLI or serving
- **THEN** settings are provided by the canonical runtime loader and no from_env helpers are used

#### Scenario: Metrics auth gating uses runtime settings
- **WHEN** /metrics authentication is evaluated
- **THEN** the decision uses settings from the runtime loader and not direct env lookups
