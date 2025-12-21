## MODIFIED Requirements
### Requirement: Non-DAG ingestion APIs are not public
Public APIs SHALL expose ingestion.engine only for tool execution (ToolService/ToolRunner) and
SHALL NOT expose non-Hamilton ingestion compute orchestration or standalone step classes.

#### Scenario: Standalone ingestion APIs are absent
- **WHEN** public ingestion APIs are listed
- **THEN** only tool execution interfaces are exposed and non-DAG workflows are absent

### Requirement: Single registry surface for discovery
Public interfaces SHALL expose Hamilton-derived OutputTarget metadata from the canonical catalog
as the single registry surface for discovery, and legacy DAG-free registries or TargetSpec lists
SHALL NOT be part of public APIs.

#### Scenario: Legacy registries are not exposed
- **WHEN** discovery registries are listed from public modules
- **THEN** only canonical catalog-backed discovery APIs are available

## Implementation Status
- Done: ingestion compute steps remain internal to Hamilton pipelines; public ingestion APIs
  expose tool execution interfaces.
- Remaining: migrate remaining non-Hamilton orchestration usage and remove legacy discovery
  registry surfaces (native TargetSpec fallbacks and DAG-free registries).
