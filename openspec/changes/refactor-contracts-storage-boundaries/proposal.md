# Change: Refactor contracts, storage boundaries, and query safety

## Why
Recent test failures exposed architectural drift: DuckDB types leaked into build modules,
contract-to-schema mappings diverged across layers, safe query helpers now raise, and
schema enumeration implicitly loads the full Hamilton DAG. We want best-in-class design
that restores strict boundaries, improves determinism and performance, and enables clean
dependency injection without monkeypatching.

## What Changes
- **BREAKING** Isolate DuckDB-specific types behind storage-owned protocols; build/export
  modules depend on duckdb-agnostic interfaces only.
- **BREAKING** Centralize schema-ID + exportability policy in a shared contract policy
  module used by both build and storage providers.
- Add safe table-key validation APIs and ensure safe_* query helpers never raise on
  invalid input or SQL injection probes.
- Split contract enumeration into schema-only and metadata-enriched paths, with lazy
  Hamilton DAG initialization for the enriched path only.
- Replace monkeypatch-driven behavior in tests with explicit settings and injected
  providers to enforce import-time safety and deterministic behavior.

## Impact
- Affected specs:
  - schema-contracts
  - storage-boundaries
  - query-safety
  - contract-resolution
  - config-injection
- Affected code:
  - src/codeintel/build/exports/
  - src/codeintel/build/schemas/
  - src/codeintel/storage/contracts/
  - src/codeintel/storage/helpers/
  - src/codeintel/storage/queries/
  - tests/build/
  - tests/storage/
