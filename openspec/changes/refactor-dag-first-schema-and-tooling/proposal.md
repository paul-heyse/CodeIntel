# Change: Refactor DAG-first schema derivation and tool execution

## Why
Schema derivation and tool execution are split across parallel systems, which risks drift,
conflicting behavior, and inconsistent provenance. We want a single DAG-first schema authority
with production inference and a single tool execution surface aligned with Hamilton targets.

## What Changes
- Make SchemaIndex/UnifiedSchemaProvider the authoritative schema source for DAG outputs,
  with production inference and derivation provenance.
- Restrict declared schemas to source-only inputs and explicit overrides; generate row bindings
  and Pandera schemas from the DAG-first registry.
- Treat inference failures as hard errors when no viable non-DAG alternative exists.
- Consolidate tool execution on ToolService/ToolRunner surfaced via BuildEnv providers.
- Route all ingestion through Hamilton native targets and integrate SCIP table ingestion.
- Unify incremental change detection with build input hashes/fingerprints for consistent skips.
- Persist change-detection deltas alongside build manifests for auditability.
- Align analytics resource loading with BuildEnv providers for consistent DI.
- Unify execution result types for ingestion and build targets.
- Converge registries on Hamilton tag/TargetSpec metadata as the single registry surface.
- Centralize row serialization on schema registry row models.
- Enforce constraint checks in order: Hamilton checks, then Pandera, then Pydantic as needed.
- **BREAKING** Remove non-DAG ingestion entrypoints and legacy tool execution helpers.

## Impact
- Affected specs: schema-contracts, contract-resolution, build-execution, config-injection,
  interface-hygiene
- Affected code: build/schemas, build/hamilton/contracts/schemas, config/datasets,
  ingestion/engine, ingestion/tracker, build/hashing, build/assets, build/providers,
  analytics/resources, core/plugins, build/hamilton/tag_index, build/hamilton/native/ingestion,
  cli/handlers
