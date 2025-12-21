## ADDED Requirements
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
