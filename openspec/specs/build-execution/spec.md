# build-execution Specification

## Purpose
TBD - created by archiving change remove-legacy-compat-code. Update Purpose after archive.
## Requirements
### Requirement: Native-only build target implementations
The build system SHALL execute targets using native Hamilton modules only, and
wrapper/template implementations or allowlists SHALL NOT be used.

#### Scenario: Plan entries are native
- **WHEN** a build plan is computed
- **THEN** each target is classified as native and no wrapper allowlist warnings occur

### Requirement: Single build result interface
Build handlers SHALL expose HamiltonBuildResult as the only build result shape and SHALL NOT
adapt results into legacy BuildResult interfaces.

#### Scenario: Build result is HamiltonBuildResult
- **WHEN** a build handler returns a result
- **THEN** the result is a HamiltonBuildResult without legacy adapter behavior

### Requirement: Build emits versioned asset catalog records only
Build execution SHALL persist asset catalog data only through versioned catalog tables
(build.asset_versions, build.run_asset_versions, build.asset_lineage) and SHALL NOT write
legacy build.assets records.

#### Scenario: Asset catalog persistence is versioned
- **WHEN** a build run persists asset catalog data
- **THEN** only versioned catalog tables are written and build.assets remains unused

