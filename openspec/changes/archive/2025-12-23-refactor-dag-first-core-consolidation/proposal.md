# Change: Refactor DAG-First Core Consolidation

## Why
The codebase now has multiple parallel registries, serialization paths, and execution
contexts that fragment the Hamilton-first model. Consolidating these into a single
DAG-first core reduces duplication, hardens boundaries, and improves extensibility
without changing functional outcomes.

## What Changes
- Introduce a unified ExecutionContext injected at all build, CLI, and serving
  entrypoints and propagated through Hamilton targets.
- Consolidate compute surfaces so graph, analytics, and ingestion logic are DAG-first
  and pure, with orchestration and I/O isolated to Hamilton materializers.
- Create a canonical RegistryService for datasets, semantic views, and exports used
  across build, storage, and serving, with Hamilton tags aligned to target modules.
- Introduce a StorageFacade to unify repository/gateway/view access for non-storage
  modules while preserving storage boundaries.
- Consolidate contract/schema compilation and row serialization into a single
  ContractService shared by build, storage, and serving.
- Unify export serialization via the core export registry and shared JSON/NDJSON
  coercion logic.
- Introduce a shared DB span emitter abstraction and remove duplicate attribute
  composition logic.
- Consolidate error taxonomy and catalog mapping across CLI/build/serving transports.
- Centralize core utility helpers (hashing, time, serialization, table-key parsing)
  and remove duplicates.

## Impact
- Affected specs: build-execution, config-injection, interface-hygiene,
  schema-contracts, export-formats, observability, error-reporting,
  storage-boundaries, serving-interfaces.
- Affected code: core runtime/config, build Hamilton targets, analytics/graphs/ingestion
  compute modules, storage gateways/repositories/views, serving registries, export
  encoders, observability tracing, and error handling layers.
