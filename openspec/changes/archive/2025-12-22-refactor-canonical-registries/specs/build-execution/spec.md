## ADDED Requirements
### Requirement: Canonical OutputTarget catalog is Hamilton-derived and cached
The system SHALL derive OutputTarget metadata via Hamilton introspection, store it in the
canonical catalog table keyed by the global catalog hash, and use the cached catalog for
CLI/spec serialization when available. Native TargetSpec lists (TARGET_SPECS) SHALL NOT be used
as a source of truth for OutputTarget metadata or graph construction.

#### Scenario: CLI reads cached target catalog
- **WHEN** a CLI command requests target metadata and the catalog hash matches a cached entry
- **THEN** the CLI uses the cached OutputTarget catalog without executing targets

#### Scenario: Cache miss regenerates target catalog
- **WHEN** no cached OutputTarget catalog matches the current catalog hash
- **THEN** Hamilton introspection regenerates the catalog and persists it

### Requirement: Unified manifest model across build and serving
Build, export, and serving layers SHALL use a single shared manifest model for run and artifact
metadata, and SHALL NOT define divergent manifest shapes.

#### Scenario: Serving export uses canonical manifest
- **WHEN** serving emits an export manifest for a dataset
- **THEN** the manifest matches the shared build/export manifest model

## Implementation Status
- Done: Hamilton-derived OutputTarget catalog generation/caching, removal of native
  TargetSpec fallbacks, and manifest consolidation are implemented. CLI/spec serialization
  reads OutputTarget metadata exclusively from the canonical catalog.
