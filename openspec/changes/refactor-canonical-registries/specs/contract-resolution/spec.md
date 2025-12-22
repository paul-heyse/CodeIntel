## RENAMED Requirements
- FROM: `### Requirement: Schema-only contract enumeration is DAG-free`
- TO: `### Requirement: Schema-only contract enumeration is catalog-backed`
- FROM: `### Requirement: Build contract resolution uses source-only providers`
- TO: `### Requirement: Build contract resolution uses canonical catalog`

## MODIFIED Requirements
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
regenerated) and SHALL NOT use a full declared schema provider for DAG outputs. Declared schemas
MAY be used only for source-only datasets or explicit overrides.

#### Scenario: DAG outputs ignore declared schemas
- **WHEN** a DAG-produced table key is resolved in build
- **THEN** its contract is sourced from the canonical catalog rather than declared schemas

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

## ADDED Requirements
### Requirement: Canonical contract catalog hash policy
The system SHALL compute a global catalog hash from Hamilton module digests, schema registry
hashes, and build configuration inputs, and SHALL use that hash to validate cached contract
catalog entries.

#### Scenario: Identical inputs yield identical catalog hashes
- **WHEN** the Hamilton modules, schema registry, and build config inputs are unchanged
- **THEN** the computed catalog hash is stable across runs

## Implementation Status
- Done: catalog hash computation, cached catalog enumeration, and contract resolution via the
  canonical catalog are in place.
- Remaining: remove declared overrides for DAG outputs (after Pandera coverage for non-inferable
  outputs) and retire legacy registry shims as CLI/spec paths move to canonical catalogs.
