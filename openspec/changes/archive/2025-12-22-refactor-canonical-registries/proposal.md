# Change: Canonical registries and runtime consolidation

## Why
The codebase currently maintains multiple sources of truth for schemas, contracts, target
metadata, row serialization, and validation. This duplication risks drift in column ordering,
contract metadata, and exported schemas, while split execution paths and configuration parsing
lead to inconsistent caching, manifests, and error handling. Consolidating on Hamilton-based
registries and shared services provides a single authoritative model, improves extensibility,
and reduces maintenance.

## What Changes
- Consolidate schema/contract resolution on the Hamilton registry; restrict declared schemas to
  source-only overrides and remove them for DAG outputs.
- Introduce a global, hash-addressed canonical catalog for DatasetContract and OutputTarget
  metadata stored in a single metadata table (DB-only) with payloads and input hashes. On hash
  mismatch, regenerate via Hamilton introspection without executing targets.
- Centralize row serialization and JSON Schema generation on the canonical schema registry; remove
  ad-hoc serializers and TypedDict-based export schemas.
- Standardize contract validation on a shared Hamilton registry validator across build, storage,
  and serving.
- Make Hamilton-derived OutputTarget metadata the single discovery surface for CLI/spec
  serialization; remove native TargetSpec fallbacks and legacy registry shims, including
  TARGET_SPECS lists used for graph construction.
- Migrate remaining ingestion/compute orchestration into Hamilton targets; keep
  ingestion.engine as tool execution only.
- Standardize resource registry access and runtime configuration loading; reduce divergent
  environment parsing.
- Re-export storage Ibis IO for Hamilton, unify manifest models, and align error taxonomy on the
  core ProblemDetail model.
- Complete Pandera/DatasetSchema registry coverage for non-inferable outputs and remove DAG
  outputs from declared_schemas, with explicit overrides living in Hamilton registry metadata.
- **BREAKING**: remove legacy registry exposure and native TargetSpec fallbacks; declared_schemas
  are source-only.

## Status Update
All scope is complete. Canonical catalogs, hash-based caching, and Hamilton introspection are the
sole sources of contract and OutputTarget metadata. Declared schemas are source-only, row
serialization and JSON Schema generation are centralized, contract validation is shared across
build/storage/serving, and runtime/resource registries are unified. Legacy registry shims and
native TargetSpec fallbacks are removed, ingestion orchestration is Hamilton-only, and quality
gates + tests are green.

## Impact
- Affected specs: schema-contracts, contract-resolution, build-execution, config-injection,
  interface-hygiene, storage-boundaries, error-reporting
- Affected code: src/codeintel/build/schemas, src/codeintel/core/schemas,
  src/codeintel/build/hamilton, src/codeintel/build/target_specs.py,
  src/codeintel/build/hamilton/driver_factory.py,
  src/codeintel/build/hamilton/nodes/support_factory.py,
  src/codeintel/config/datasets, src/codeintel/storage/metadata,
  src/codeintel/ingestion, src/codeintel/build/analytics/compute/hotspots,
  src/codeintel/core/errors, src/codeintel/serving/errors,
  src/codeintel/build/exports/manifest.py, src/codeintel/build/serving/manifest.py,
  src/codeintel/build/analytics_resources.py, src/codeintel/build/providers.py
