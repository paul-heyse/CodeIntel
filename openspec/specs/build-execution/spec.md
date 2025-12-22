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
run standalone ingestion steps or helper services (for example, BuildToolAdapter or
ChangeTracker) outside the DAG.

#### Scenario: Ingestion targets are Hamilton-native
- **WHEN** ingestion is invoked for a snapshot
- **THEN** the execution plan contains Hamilton-native targets and no standalone ingestion
  entrypoints, BuildToolAdapter, or ChangeTracker services are called

### Requirement: Canonical tool execution service
Build targets SHALL execute external tools via ToolService/ToolRunner provided by BuildEnv
providers, and duplicate tool runner implementations (including native tool executor
helpers in build.hamilton.native.tools) SHALL NOT be used or exported.

#### Scenario: Build targets use ToolService
- **WHEN** a build target executes an external tool
- **THEN** the target uses ToolService from BuildEnv providers rather than a private runner
  or native tool executor helpers

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

### Requirement: Legacy execution result helpers are removed
Build execution SHALL expose only canonical result types and helper methods, and SHALL NOT
provide compatibility aliases or custom ResultBuilder adapters for legacy result shapes.

#### Scenario: Execution results use canonical helpers
- **WHEN** build or ingestion results are constructed or imported
- **THEN** only HamiltonBuildResult and ExecutionResult.ok/failed/skip are used and no
  BuildResultBuilder, DictResultBuilder, or ExecutionResult.fail alias is exposed

### Requirement: Semantic registry compilation uses canonical compiler
Semantic registry compilation SHALL use build.serving.semantic_compile and SHALL NOT rely
on alternate tag-discovery compile helpers.

#### Scenario: Semantic registry uses canonical compiler
- **WHEN** semantic registry artifacts are compiled
- **THEN** build.serving.semantic_compile is used and semantic_compile_hamilton is not present

### Requirement: Canonical OutputTarget catalog is Hamilton-derived and cached
The system SHALL derive OutputTarget metadata via Hamilton introspection, store it in the
canonical catalog table keyed by the global catalog hash, and use the cached catalog for
CLI/spec serialization when available. Native TargetSpec lists (TARGET_SPECS) SHALL NOT be used
as a source of truth for OutputTarget metadata or graph construction.

#### Scenario: CLI reads cached target catalog
- **WHEN** a CLI command requests target metadata and the catalog hash matches a cached entry
- **THEN** the CLI uses the cached OutputTarget catalog without executing targets

#### Scenario: Cache miss regenerates target catalog
- **WHEN** no cached OutputTarget catalog matches the current catalog hash
- **THEN** Hamilton introspection regenerates the catalog and persists it

### Requirement: Unified manifest model across build and serving
Build, export, and serving layers SHALL use a single shared manifest model for run and artifact
metadata, and SHALL NOT define divergent manifest shapes.

#### Scenario: Serving export uses canonical manifest
- **WHEN** serving emits an export manifest for a dataset
- **THEN** the manifest matches the shared build/export manifest model

