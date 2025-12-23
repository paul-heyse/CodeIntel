## 1. Architecture Consolidation
- [x] 1.1 Define and wire a unified ExecutionContext for build, CLI, and serving entrypoints.
- [ ] 1.2 Consolidate compute modules into DAG-first, pure transforms with Hamilton
      materializers as the only write path. (Remaining: finish migrating compute modules
      to pure transforms; isolate writes to materializers.)
- [x] 1.3 Implement a canonical RegistryService for datasets, semantic views, and exports;
      migrate all discovery and catalog callers.
- [x] 1.4 Introduce a StorageFacade for non-storage modules to access read/write/export
      operations through a single API.
- [x] 1.5 Consolidate contract/schema compilation into a single ContractService and
      standardize row serialization and validation.
- [x] 1.6 Unify export serialization paths via the core export registry and shared
      JSON/NDJSON coercion logic.
- [x] 1.7 Introduce a shared DB span emitter abstraction and refactor DuckDB tracing
      to use it.
- [x] 1.8 Unify error taxonomy and catalog mapping across CLI/build/serving transports.
- [ ] 1.9 Centralize hashing/time/serialization helpers in core modules and remove
      duplicate utility implementations. (Remaining: retire remaining duplicate helpers.)

## 2. Migration and Cleanup
- [ ] 2.1 Remove legacy registries, duplicate serializers, and redundant helper modules.
      (Remaining: remove legacy registries/serializers once unused.)
- [x] 2.2 Update integration points (CLI/build/serving) to use the new services and
      execution context.
- [ ] 2.3 Update Hamilton tags/metadata to align with the consolidated registry service.
      (Remaining: align tag metadata with RegistryService outputs.)

## 3. Tests and Validation
- [x] 3.1 Update unit tests for registry discovery, contract compilation, and exports.
- [x] 3.2 Update storage facade and observability tests to match the consolidated APIs.
- [x] 3.3 Run `uv run python -m tools.quality_report --output build/quality-results/
      quality_report.json` and `uv run pytest -q`.
