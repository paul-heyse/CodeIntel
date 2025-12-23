## ADDED Requirements
### Requirement: ExecutionContext is the canonical DAG input
Hamilton targets SHALL accept a unified ExecutionContext that includes snapshot identity,
settings, runtime primitives, and run metadata. Build, CLI, and serving entrypoints SHALL
construct and inject this context instead of passing ad-hoc parameter bundles.

#### Scenario: DAG targets receive the execution context
- **WHEN** a build plan is executed for a snapshot
- **THEN** each Hamilton target receives a shared ExecutionContext with snapshot and
  runtime settings

### Requirement: DAG-first compute modules are pure and isolated
Graph, analytics, and ingestion compute modules SHALL remain pure transforms that return
row data or intermediate structures and SHALL NOT perform writes or orchestration. All
materialization and side effects SHALL be handled by Hamilton materializers.

#### Scenario: Compute module avoids direct writes
- **WHEN** a compute module is invoked by a Hamilton target
- **THEN** it performs no direct storage writes and returns data for materialization
