# Change: Refactor DAG-First Consolidation Across Build/Storage/Serving/Core

## Why
The current build/storage/serving layers still duplicate contract derivation, export format
registries, and error payload models, and some build utilities reach into DuckDB-specific APIs.
These seams break the storage boundary, risk drift between layers, and trigger Hamilton DAG
initialization in schema-only code paths that are intended to be DAG-free.

## What Changes
- Consolidate dataset contract derivation into a shared core factory used by both build and
  storage providers, including view detection, owner mapping, row bindings, and export defaults.
- Introduce a storage-owned export surface (relation building + audit logging) so build exports
  rely only on duckdb-agnostic protocols.
- Provide DAG-free output inventory and contract enumeration paths that never initialize the
  Hamilton driver unless metadata enrichment is explicitly requested.
- Move environment resolution to boundary loaders and pass explicit settings objects through
  BuildEnv/ServingRuntime/ConfigRegistry instead of implicit `from_env` or global getters.
- Unify error payloads around the core RFC 9457 ProblemDetail model with serving adapters, and
  align error code mapping with stable catalog entries.
- Create a canonical export format registry (with alias handling for jsonl/ndjson) shared by
  build and serving surfaces.

## Impact
- Affected specs: storage-boundaries, schema-contracts, contract-resolution, config-injection,
  error-reporting (new), export-formats (new).
- Affected code:
  - Build: `src/codeintel/build/exports/*`, `src/codeintel/build/schemas/contract_service.py`,
    `src/codeintel/build/schemas/provider_declared.py`, `src/codeintel/build/run_context.py`,
    `src/codeintel/build/settings.py`.
  - Storage: `src/codeintel/storage/contracts/provider.py`, `src/codeintel/storage/gateway/protocol.py`,
    `src/codeintel/storage/duckdb_policy_backend.py`, `src/codeintel/storage/datasets/registry.py`.
  - Serving: `src/codeintel/serving/http/errors.py`, `src/codeintel/serving/errors/*`,
    `src/codeintel/serving/export/*`, `src/codeintel/serving/settings.py`.
  - Core: `src/codeintel/core/schemas/*`, `src/codeintel/core/errors/*`, `src/codeintel/core/config/*`.
