## MODIFIED Requirements
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

## ADDED Requirements
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
