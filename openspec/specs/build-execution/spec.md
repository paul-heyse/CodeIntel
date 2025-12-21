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

### Requirement: Hamilton-native ingestion execution
The build system SHALL execute ingestion workflows as Hamilton native targets and SHALL NOT
run standalone ingestion steps outside the DAG.

#### Scenario: Ingestion targets are Hamilton-native
- **WHEN** ingestion is invoked for a snapshot
- **THEN** the execution plan contains Hamilton-native targets and no standalone ingestion
  entrypoints are called

### Requirement: Canonical tool execution service
Build targets SHALL execute external tools via ToolService/ToolRunner provided by BuildEnv
providers, and duplicate tool runner implementations SHALL NOT be used.

#### Scenario: Build targets use ToolService
- **WHEN** a build target executes an external tool
- **THEN** the target uses ToolService from BuildEnv providers rather than a private runner

### Requirement: Unified change detection and hashing
Incremental ingestion change detection SHALL be aligned with build input hashing and
fingerprint policy so that skip/rebuild decisions use a single authoritative hash source.

#### Scenario: Skip decisions use a unified hash authority
- **WHEN** ingestion and build targets evaluate whether to skip computation
- **THEN** both use the same input hash or fingerprint data to decide

### Requirement: Change-detection deltas are auditable
Change-detection deltas SHALL be persisted alongside build manifests to support auditability
of skip/rebuild decisions.

#### Scenario: Change deltas are stored with manifests
- **WHEN** change detection computes added, modified, and deleted paths
- **THEN** the deltas are stored with the corresponding build manifest records

### Requirement: Unified execution result model
Ingestion targets SHALL emit execution results using the same shared result model as
executor-style build targets, including skip semantics and structured errors.

#### Scenario: Ingestion result uses shared model
- **WHEN** an ingestion target completes or is skipped
- **THEN** the result is represented by the shared execution result model

