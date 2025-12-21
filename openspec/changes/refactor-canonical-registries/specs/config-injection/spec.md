## MODIFIED Requirements
### Requirement: Analytics resources use injected registry access
Analytics and graph resource loading SHALL use a single registry interface supplied by
BuildEnv/Providers, and modules SHALL NOT construct standalone registry implementations.

#### Scenario: Analytics registry comes from providers
- **WHEN** analytics or graph code requires access to the resource registry
- **THEN** it uses the injected registry interface from providers rather than constructing one

## ADDED Requirements
### Requirement: Canonical runtime configuration loader
The system SHALL provide a single runtime configuration loader that returns RuntimePrimitives
and settings for build, serving, and CLI entrypoints, and environment parsing SHALL be confined
to that loader.

#### Scenario: Runtime loader centralizes environment parsing
- **WHEN** a CLI entrypoint constructs runtime primitives
- **THEN** it uses the canonical loader and no library modules read environment variables

## Implementation Status
- Done: a unified runtime loader now returns RuntimePrimitives plus settings for build/serving/CLI.
- Remaining: standardize the BuildEnv resource registry interface with core ResourceRegistry and
  remove standalone registry construction in analytics/graphs.
