## ADDED Requirements

### Requirement: DAG-first schema authority
The system SHALL derive schemas for all DAG-produced tables and views from the global Hamilton DAG.
Declared schemas SHALL be used only for source tables and explicit overrides.

#### Scenario: DAG output schema resolution
- **WHEN** a table_key is produced by the global DAG
- **THEN** SchemaService returns the schema derived from SchemaIndex without consulting declared
  source registries

#### Scenario: Source table schema resolution
- **WHEN** a table_key is not produced by the global DAG
- **THEN** SchemaService resolves the schema from the declared source schema registry

### Requirement: Manifest-backed SchemaService for runtime
Runtime layers (storage, serving, CLI) SHALL resolve schemas exclusively from SchemaManifest
artifacts via a manifest-backed SchemaService, and MUST NOT import build modules.

#### Scenario: Serving loads schema from artifact
- **WHEN** serving initializes SchemaService
- **THEN** it loads SchemaManifest from the artifact provider and resolves schemas from it

#### Scenario: Missing schema manifest
- **WHEN** SchemaService is requested but SchemaManifest is unavailable
- **THEN** the system returns a ProblemDetail error indicating missing artifacts

### Requirement: Canonical JSON Schema generation
JSON Schemas SHALL be generated via the core JSON Schema generator from TableSchema inputs using
JSON Schema draft 2020-12, and their digests SHALL be computed from canonical JSON serialization.

#### Scenario: JSON Schema generation
- **WHEN** a TableSchema is provided
- **THEN** the generated JSON Schema declares draft 2020-12 and includes required fields

#### Scenario: Stable JSON Schema digest
- **WHEN** the same TableSchema is provided across runs
- **THEN** the JSON Schema digest is identical

### Requirement: Schema provenance in manifests
SchemaManifest entries SHALL include schema_hash and derivation metadata (kind and source) for all
tables and views.

#### Scenario: Table provenance
- **WHEN** a SchemaManifest is written
- **THEN** each table entry includes schema_hash, derivation_kind, and derivation_source

#### Scenario: View derivation kind
- **WHEN** a view schema is inferred from SQL or DAG lineage
- **THEN** derivation_kind is recorded as view_inferred
