## MODIFIED Requirements
### Requirement: Non-DAG ingestion APIs are not public
Public APIs SHALL expose ingestion.engine only for tool execution (ToolService/ToolRunner)
and SHALL NOT expose non-Hamilton ingestion, analytics, graph, or history compute
orchestration or standalone step classes. CLI/debug entrypoints SHALL route through
DAG-derived outputs rather than direct compute modules.

#### Scenario: Standalone orchestration APIs are absent
- **WHEN** public ingestion/analytics/graph APIs are enumerated
- **THEN** only tool execution interfaces are exposed and no non-DAG orchestration is present

#### Scenario: CLI debug uses DAG outputs
- **WHEN** a CLI debug or analytics command is invoked
- **THEN** it reads DAG-produced datasets or triggers DAG targets rather than running
  module-level orchestration

## ADDED Requirements
### Requirement: Canonical ID normalization utilities
ID normalization SHALL use canonical helpers from codeintel.core.data_models.ids, and
packages SHALL NOT introduce duplicate ID conversion utilities for graph/analytics IDs.

#### Scenario: Graph metrics use canonical ID helpers
- **WHEN** graph or analytics code normalizes IDs for output rows
- **THEN** it uses codeintel.core.data_models.ids helpers and no duplicate converters exist
