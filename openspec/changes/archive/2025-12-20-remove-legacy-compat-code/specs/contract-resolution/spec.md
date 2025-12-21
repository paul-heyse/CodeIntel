## ADDED Requirements
### Requirement: Build contract resolution uses source-only providers
Build-layer contract enumeration SHALL use source-only declared schema providers and
SHALL NOT expose a full declared schema provider from build APIs.

#### Scenario: Schema-only enumeration excludes DAG outputs
- **WHEN** build contract enumeration runs in schema-only mode
- **THEN** DAG-produced table keys are excluded and no full provider is available from build
