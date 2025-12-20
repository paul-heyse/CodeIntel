## ADDED Requirements

### Requirement: Build artifacts are first-class outputs
The build system SHALL emit SchemaManifest, DatasetCatalog, and SemanticRegistry artifacts as
first-class outputs in a deterministic location with stable filenames and version metadata.

#### Scenario: Artifact emission
- **WHEN** a build completes successfully
- **THEN** `build/artifacts/schema_manifest.json`, `build/artifacts/dataset_catalog.json`, and
  `build/artifacts/semantic_registry.json` are written with version fields

#### Scenario: Deterministic artifact ordering
- **WHEN** the same DAG and inputs are built twice
- **THEN** the artifact contents are byte-identical

### Requirement: Artifact provider API
The system SHALL provide an ArtifactProvider API that loads, validates, and caches build artifacts
for runtime use.

#### Scenario: Artifact loading
- **WHEN** runtime initializes the ArtifactProvider
- **THEN** it validates artifact schemas and exposes accessors for manifest and catalog data

#### Scenario: Missing artifact handling
- **WHEN** an artifact is missing or invalid
- **THEN** the provider returns a ProblemDetail error describing the failure

### Requirement: Runtime layers use artifact boundary
Storage and serving layers SHALL consume schemas, contracts, and semantic metadata exclusively via
ArtifactProvider outputs.

#### Scenario: Storage schema resolution
- **WHEN** storage resolves a table schema
- **THEN** it reads from SchemaManifest via ArtifactProvider rather than declared registries

#### Scenario: Serving semantic registry resolution
- **WHEN** serving lists semantic views
- **THEN** it loads the SemanticRegistry artifact via ArtifactProvider
