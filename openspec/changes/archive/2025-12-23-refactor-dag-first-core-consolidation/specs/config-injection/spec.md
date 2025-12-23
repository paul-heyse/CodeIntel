## ADDED Requirements
### Requirement: ExecutionContext is built by the runtime loader
The system SHALL construct ExecutionContext via the canonical runtime configuration
loader and SHALL NOT build it ad-hoc within modules. Environment parsing SHALL be
confined to the runtime loader and the explicit CLI env override allowlist.

#### Scenario: Entry points use the runtime loader
- **WHEN** a CLI, build, or serving entrypoint starts
- **THEN** it uses the canonical runtime loader to construct ExecutionContext
  without module-level environment parsing
