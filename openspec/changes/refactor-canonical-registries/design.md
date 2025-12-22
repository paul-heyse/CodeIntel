## Context
Schema contracts, target metadata, row serialization, and validation are currently derived from
multiple registries and helper modules. Some enumeration paths are DAG-free while others depend
on Hamilton introspection, which leads to drift between build, storage, and serving. Execution
surfaces also split between Hamilton and legacy ingestion/compute orchestration, and IO/error/
manifest concerns are duplicated across layers.

## Goals / Non-Goals
- Goals:
  - Use Hamilton registry outputs as the single source of truth for DatasetContract and
    OutputTarget metadata.
  - Provide a global, hash-addressed catalog cache stored in DuckDB metadata for fast
    enumeration without target execution.
  - Centralize row serialization, contract validation, and JSON Schema generation on the
    schema registry.
  - Make ingestion/compute orchestration Hamilton-only while keeping tooling execution available.
  - Align runtime configuration, resource registries, IO adapters, error taxonomy, and
    manifest models across layers.
- Non-Goals:
  - Change dataset semantics or schema content beyond consolidation.
  - Introduce new export formats or serving APIs.
  - Replace Hamilton execution model.

## Decisions
- Use Hamilton DAG introspection (no node execution) to generate catalogs for contracts and
  OutputTargets.
- Store canonical catalogs in a single metadata table keyed by catalog_kind + global catalog_hash
  with DB-only persistence (payload + input hashes stored for traceability).
- Compute catalog_hash from Hamilton module digests, schema registry hashes, and build config
  inputs, scoped globally (not per repo).
- Build catalog payloads via ContractService/TargetMetadataService and persist on cache miss.
- Centralize row serialization and JSON Schema generation in schema-registry-backed services,
  and replace ad-hoc serializer helpers.
- Consolidate contract validation into a shared validator invoked by build, storage, and serving.
- Restrict declared_schemas to source-only datasets and explicit overrides; remove usage for
  DAG outputs in target metadata and schema resolution paths.
- Eliminate TARGET_SPECS/native target enumeration; derive OutputTarget metadata directly from
  Hamilton node metadata and canonical catalogs.
- Complete Pandera/DatasetSchema coverage for non-inferable outputs and remove DAG outputs from
  declared_schemas; allow TargetSpecOptions overrides only as a temporary bridge.
- Provide a single runtime configuration loader that returns RuntimePrimitives and settings
  for build/serving/CLI.
- Re-export storage Ibis IO for Hamilton to enforce consistent Ibis 11 patterns.
- Unify manifest models and error taxonomy around core ProblemDetail and ErrorCode.
- Treat DAG-free contract enumeration as a catalog-backed accessor only; legacy registry
  surfaces should be removed once CLI/spec paths rely on the canonical catalogs.

## Alternatives considered
- Keep DAG-free contract enumeration and accept drift between registries.
- Use file-based catalogs for contract/target metadata instead of database storage.
- Maintain per-repo catalog hashing keyed by snapshot rather than global configuration inputs.

## Risks / Trade-offs
- Catalog hash invalidation may be too sensitive or too coarse; improper inputs could cause
  stale catalogs or frequent regenerations.
- Removing DAG-free target spec enumeration may introduce startup cost from Hamilton
  introspection on cache misses.
- Transitioning off declared_schemas requires complete Pandera registry coverage; early removal
  could break source-only datasets.
- Removing TARGET_SPECS requires consistent Hamilton node metadata to preserve OutputTarget
  descriptions; missing metadata will degrade catalog fidelity.

## Migration Plan
1. Add metadata table for canonical catalogs and implement storage APIs for load/store. (Done)
2. Implement catalog hash computation and Hamilton introspection catalog generator. (Done)
3. Update contract resolution to use cached catalogs, falling back to introspection on hash
   mismatch, and restrict declared_schemas to source-only datasets. (Partial)
4. Replace row serialization helpers with a centralized schema registry serializer and caching.
   (Done)
5. Replace DAG-free target catalog generation with canonical OutputTarget catalog consumption,
   remove TARGET_SPECS usage, and derive OutputTarget metadata from Hamilton node tags.
   (Remaining)
6. Introduce unified runtime configuration loader and consolidate resource registry interfaces.
   (Partial)
7. Re-export storage Ibis IO for Hamilton, unify manifest models, and align error taxonomy.
   (Done)
8. Remove deprecated registry/contract helpers and update tests/quality gates. (Remaining)
9. Migrate remaining non-Hamilton ingestion/analytics orchestration into Hamilton nodes.
   (Remaining)

## Implementation Status
Completed work includes canonical catalog hashing/storage, Hamilton catalog generation, cached
contract resolution, centralized row serialization, JSON Schema generation, shared contract
validation, unified runtime loader, storage-backed Ibis IO, manifest consolidation, and error
taxonomy alignment.

Remaining scope includes removing native TargetSpec fallbacks, finishing declared_schemas
restrictions for DAG outputs, unifying resource registry interfaces in BuildEnv, cleaning legacy
registry shims, migrating non-Hamilton orchestration usage, and completing quality gates/tests.
Remaining scope also includes completing Pandera/DatasetSchema coverage for non-inferable
outputs and removing DAG output entries from declared_schemas.

## Open Questions
- None.
