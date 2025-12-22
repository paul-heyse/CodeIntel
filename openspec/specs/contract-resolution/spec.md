# contract-resolution Specification

## Purpose
TBD - created by archiving change refactor-contracts-storage-boundaries. Update Purpose after archive.
## Requirements
### Requirement: Lazy metadata enrichment
Metadata enrichment SHALL be lazy and only initialize the Hamilton DAG when explicitly
requested via injected metadata providers.

#### Scenario: Metadata requested triggers DAG initialization
- **WHEN** metadata enrichment is requested for a contract
- **THEN** the Hamilton DAG initializes to provide the metadata

### Requirement: Injectable metadata providers
The system SHALL allow dependency injection of metadata and output inventory providers to
support testability and alternative runtime implementations.

#### Scenario: Tests inject a stub metadata provider
- **WHEN** a test supplies a stub metadata provider and output inventory
- **THEN** contract enumeration succeeds without DAG initialization

### Requirement: DAG-first schema provider with production inference
The system SHALL expose a DAG-first schema provider that resolves DAG outputs via SchemaIndex,
with production inference enabled and derivation provenance recorded; callers SHALL be able to
disable inference explicitly.

#### Scenario: Inference resolves a DAG output schema
- **WHEN** a schema is requested for an inferable DAG-produced table key
- **THEN** the provider returns the inferred schema with provenance marked as inferred

#### Scenario: Inference can be disabled
- **WHEN** a caller requests schemas with inference disabled
- **THEN** the provider returns explicit overrides or None without triggering inference

### Requirement: Inference failures are hard errors when no alternative exists
The system SHALL raise a hard error when DAG schema inference fails and no viable non-DAG
alternative schema is available for the dataset.

#### Scenario: Inference failure without fallback is fatal
- **WHEN** schema inference fails for a DAG-produced table key without a fallback override
- **THEN** schema resolution fails with a hard error

### Requirement: Contract resolution defaults to DAG-first outputs
Contract resolution SHALL default to canonical catalog entries that include DAG outputs with
metadata overrides applied. A declared-only mode MAY be provided for source-only enumeration and
SHALL exclude DAG outputs.

#### Scenario: Default resolution includes DAG outputs
- **WHEN** a caller resolves contracts without specifying a resolution mode
- **THEN** DAG-produced table keys are included via the canonical catalog

#### Scenario: Declared-only mode excludes DAG outputs
- **WHEN** a caller requests declared-only contract resolution
- **THEN** DAG-produced table keys are excluded from the results

### Requirement: Schema-only contract enumeration is catalog-backed
Schema-only contract enumeration SHALL be served from the canonical contract catalog stored in
metadata and keyed by the global catalog hash. Enumeration MAY initialize the Hamilton DAG for
introspection-only regeneration on cache miss, and SHALL NOT execute targets.

#### Scenario: Cached catalog served without execution
- **WHEN** the canonical catalog hash matches a stored catalog entry
- **THEN** enumeration uses the cached catalog without executing targets

#### Scenario: Cache miss regenerates via introspection
- **WHEN** no cached catalog matches the current catalog hash
- **THEN** Hamilton introspection regenerates the catalog and persists it for reuse

### Requirement: Build contract resolution uses canonical catalog
Build-layer contract resolution SHALL resolve contracts from the canonical catalog (cached or
regenerated) and SHALL NOT use declared_schemas for DAG outputs. Declared schemas SHALL be used
only for source-only datasets, and explicit overrides SHALL be sourced from Hamilton registry
metadata.

#### Scenario: DAG outputs ignore declared schemas
- **WHEN** a DAG-produced table key is resolved in build
- **THEN** its contract is sourced from the canonical catalog rather than declared schemas

### Requirement: Canonical contract catalog hash policy
The system SHALL compute a global catalog hash from Hamilton module digests, schema registry
hashes, and build configuration inputs, and SHALL use that hash to validate cached contract
catalog entries.

#### Scenario: Identical inputs yield identical catalog hashes
- **WHEN** the Hamilton modules, schema registry, and build config inputs are unchanged
- **THEN** the computed catalog hash is stable across runs

