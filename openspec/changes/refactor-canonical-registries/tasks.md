## 1. Catalog Persistence and Hashing
- [x] 1.1 Add a metadata table to store canonical catalogs by kind and global hash
- [x] 1.2 Define a global catalog hash computation from Hamilton module digests, schema registry
      hashes, and build config inputs
- [x] 1.3 Implement storage APIs to read/write catalog payloads (DB-only)

## 2. Hamilton Catalog Generation
- [x] 2.1 Implement Hamilton introspection catalog generation for DatasetContract metadata
- [x] 2.2 Implement Hamilton introspection catalog generation for OutputTarget metadata
- [x] 2.3 Wire catalog regeneration on hash mismatch without executing targets

## 3. Contract Resolution Consolidation
- [x] 3.1 Update contract resolution to use cached canonical catalogs by default
- [ ] 3.2 Restrict declared schemas to source-only datasets and explicit overrides
      (remaining: complete Pandera/DatasetSchema registry coverage for non-inferable outputs,
      remove DAG output entries from declared_schemas, and allow explicit overrides only via
      Hamilton registry metadata or temporary TargetSpecOptions bridges)
- [x] 3.3 Consolidate JSON Schema generation on contract/Pandera-derived schemas
- [x] 3.4 Introduce a shared contract validation service used by build, storage, and serving

## 4. Row Serialization Consolidation
- [x] 4.1 Build a centralized schema-registry-backed row serializer with caching
- [x] 4.2 Update build/ingestion/validation call sites to use the centralized serializer
- [x] 4.3 Remove ad-hoc row serialization helpers that duplicate ordering logic

## 5. Target Catalog Consolidation
- [ ] 5.1 Remove DAG-free TargetSpec catalog enumeration paths
      (remaining: delete TARGET_SPECS/load_native_target_specs usage in driver/support and derive
      OutputTarget metadata + graph from Hamilton introspection and canonical catalogs)
- [ ] 5.2 Update CLI/spec serialization to read OutputTarget metadata from the canonical catalog
      (remaining: verify remaining call sites, docs, and examples use canonical catalog only)

## 6. Runtime and Resource Registry Consolidation
- [x] 6.1 Implement a unified runtime configuration loader returning RuntimePrimitives + settings
- [ ] 6.2 Standardize resource registry interfaces across core/analytics/graphs and BuildEnv
      (remaining: remove build-only registry wrapper and use core ResourceRegistry +
      ProviderFactory consistently)

## 7. IO, Errors, and Manifests
- [x] 7.1 Re-export storage Ibis IO for Hamilton and remove duplicate adapters
- [x] 7.2 Align build/serving error taxonomy with core ProblemDetail and error codes
- [x] 7.3 Consolidate manifest models used by build, export, and serving layers

## 8. Cleanup and Validation
- [ ] 8.1 Remove deprecated registry helpers and declared_schemas usage for DAG outputs
      (remaining: remove DAG-free registry shims such as target_specs and declared DAG outputs,
      and ensure schema resolution has no declared fallback for DAG tables)
- [ ] 8.2 Update tests and documentation to reflect canonical registries
      (remaining: full doc/test sweep beyond touched files)
- [ ] 8.3 Run quality_report and pytest gates
      (remaining: guardrails and pytest timeouts)

## 9. Ingestion and Interface Hygiene
- [ ] 9.1 Migrate remaining non-Hamilton ingestion/analytics orchestration into Hamilton nodes
      (remaining: migrate hotspots compute and any remaining tool-runner orchestration in
      analytics modules into Hamilton)
- [ ] 9.2 Ensure ingestion public APIs expose tool execution only (confirm __all__ and docs)
