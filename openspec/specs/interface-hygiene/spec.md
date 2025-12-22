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
CLI surfaces SHALL NOT expose deprecated or no-op commands such as storage.generate_macros,
and SHALL NOT ship standalone CLI entrypoint modules that are not registered in the
primary CLI applications (for example, legacy completion installers, background job
runners, skip-arg helper modules, or module-level MCP entrypoints).

#### Scenario: Deprecated storage command is absent
- **WHEN** storage CLI commands are listed
- **THEN** storage.generate_macros is not present

#### Scenario: Legacy CLI entrypoints are absent
- **WHEN** CLI entrypoints are enumerated for distribution
- **THEN** legacy completion installers, job runner modules, skip-arg helpers, and
  module-level MCP entrypoints are not shipped

### Requirement: Non-DAG ingestion APIs are not public
Public APIs SHALL NOT expose standalone ingestion step classes or helper services such as
BuildToolAdapter or ChangeTracker, and ingestion SHALL be invoked only through Hamilton
targets.

#### Scenario: Standalone ingestion APIs are absent
- **WHEN** public ingestion APIs are listed
- **THEN** non-DAG ingestion step classes and helper services (BuildToolAdapter,
  ChangeTracker) are not exposed

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

### Requirement: Manifest compatibility shims are removed
Public build and ingestion packages SHALL NOT re-export manifest helpers, and callers
MUST import ManifestBase, read_manifest_json, and write_manifest_json from
codeintel.core.manifests.

#### Scenario: Manifest helpers use canonical module
- **WHEN** manifest helpers are imported by build or ingestion code
- **THEN** they come from codeintel.core.manifests and no build.manifest_* shims exist

### Requirement: Architecture boundary checks are test-only
Architecture boundary enforcement SHALL be test-only and SHALL NOT be shipped as a runtime
package module.

#### Scenario: Runtime packages exclude architecture test helpers
- **WHEN** runtime package modules are enumerated
- **THEN** codeintel._architecture is not present and boundary checks live in tests

