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
  source-only overrides and retire them for DAG outputs after Pandera registry coverage is
  complete.
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
  outputs from declared_schemas, using explicit overrides only as a temporary bridge.
- **BREAKING**: remove legacy registry exposure and native TargetSpec fallbacks; declared_schemas
  usage for DAG outputs will be removed after Pandera coverage is complete.

## Status Update
Completed work includes canonical catalog hashing/storage, Hamilton catalog generation, cached
contract resolution, centralized row serialization, contract-derived JSON Schema, shared
contract validation, unified runtime loader, storage-backed Ibis IO, manifest consolidation, and
error taxonomy alignment.

Remaining scope includes removing TargetSpec/native fallback paths, fully restricting
declared_schemas for DAG outputs (with Pandera coverage for non-inferable outputs), unifying
BuildEnv resource registry interfaces, migrating non-Hamilton orchestration usage (e.g., hotspots),
cleaning legacy registry shims, and completing quality gates and tests.

## Impact
- Affected specs: schema-contracts, contract-resolution, build-execution, config-injection,
  interface-hygiene, storage-boundaries, error-reporting
- Affected code: src/codeintel/build/schemas, src/codeintel/core/schemas,
  src/codeintel/build/hamilton, src/codeintel/build/target_specs.py,
  src/codeintel/build/hamilton/driver_factory.py, src/codeintel/build/hamilton/nodes/support_factory.py,
  src/codeintel/config/datasets, src/codeintel/storage/metadata, src/codeintel/ingestion,
  src/codeintel/analytics/compute/hotspots, src/codeintel/core/errors,
  src/codeintel/serving/errors, src/codeintel/build/exports/manifest.py,
  src/codeintel/build/serving/manifest.py, src/codeintel/build/analytics_resources.py,
  src/codeintel/build/providers.py
