## MODIFIED Requirements
### Requirement: Canonical tool execution service
Build targets SHALL execute external tools via ToolService/ToolRunner provided by BuildEnv
providers, and duplicate tool runner implementations (including native tool executor
helpers in build.hamilton.native.tools) SHALL NOT be used or exported.

#### Scenario: Build targets use ToolService
- **WHEN** a build target executes an external tool
- **THEN** the target uses ToolService from BuildEnv providers rather than a private runner
  or native tool executor helpers

### Requirement: Hamilton-native ingestion execution
The build system SHALL execute ingestion workflows as Hamilton native targets and SHALL NOT
run standalone ingestion steps or helper services (for example, BuildToolAdapter or
ChangeTracker) outside the DAG.

#### Scenario: Ingestion targets are Hamilton-native
- **WHEN** ingestion is invoked for a snapshot
- **THEN** the execution plan contains Hamilton-native targets and no standalone ingestion
  entrypoints, BuildToolAdapter, or ChangeTracker services are called

## ADDED Requirements
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
