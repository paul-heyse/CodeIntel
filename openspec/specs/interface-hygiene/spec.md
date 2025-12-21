# interface-hygiene Specification

## Purpose
TBD - created by archiving change refactor-asset-catalog-and-retire-compat-shims. Update Purpose after archive.
## Requirements
### Requirement: Public interfaces exclude compatibility-only parameters
Public APIs SHALL NOT include unused compatibility-only parameters or methods. Parameters
MUST exist only when required for behavior or documented interfaces.

#### Scenario: Compatibility parameters are removed
- **WHEN** public APIs are reviewed for unused parameters
- **THEN** compatibility-only parameters are removed and call sites are updated

### Requirement: Deprecated or no-op commands are not exposed
CLI surfaces SHALL NOT expose deprecated or no-op commands such as storage.generate_macros.

#### Scenario: Deprecated storage command is absent
- **WHEN** storage CLI commands are listed
- **THEN** storage.generate_macros is not present

### Requirement: Non-DAG ingestion APIs are not public
Public APIs SHALL NOT expose standalone ingestion step classes or functions, and ingestion
SHALL be invoked only through Hamilton targets.

#### Scenario: Standalone ingestion APIs are absent
- **WHEN** public ingestion APIs are listed
- **THEN** non-DAG ingestion step classes and functions are not exposed

### Requirement: Single tool execution surface
Public interfaces SHALL expose ToolService-based execution only, and legacy build provider
helpers for tool execution SHALL NOT be part of public APIs.

#### Scenario: Legacy tool helpers are absent
- **WHEN** public tool execution APIs are listed
- **THEN** only ToolService-based interfaces are available

### Requirement: Single registry surface for discovery
Public interfaces SHALL expose Hamilton tag and TargetSpec metadata as the single registry
surface for discovery, and legacy registry implementations SHALL NOT be part of public APIs.

#### Scenario: Legacy registries are not exposed
- **WHEN** discovery registries are listed from public modules
- **THEN** only Hamilton tag/TargetSpec-based discovery APIs are available

### Requirement: No long-term compatibility facades
Public interfaces SHALL NOT expose long-term compatibility facades for analytics resource
registries, and any migration shims SHALL be removed by the end state.

#### Scenario: Resource registry compatibility facade is absent
- **WHEN** analytics resource APIs are listed
- **THEN** no compatibility facade for legacy registries is available

