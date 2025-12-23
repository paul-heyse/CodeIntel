## ADDED Requirements
### Requirement: ContractService is the single contract pipeline
The system SHALL provide a ContractService that compiles Pandera schemas, JSON Schema,
row serializers, and validation policies from DatasetContract definitions. Build,
storage, and serving layers SHALL rely on this service and SHALL NOT maintain parallel
schema compilation or serialization pipelines.

#### Scenario: Build and serving share contract compilation
- **WHEN** a dataset contract is compiled for build and serving
- **THEN** both layers use ContractService outputs with identical schemas and
  serialization behavior
