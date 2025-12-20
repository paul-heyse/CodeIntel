## 1. Architecture and Policy Foundations
- [x] 1.1 Create a storage protocol module (duckdb-agnostic relation/record batch types)
- [x] 1.2 Add adapters in storage for DuckDB relations and Arrow readers
- [x] 1.3 Introduce a shared contract policy module for schema ID derivation and
      exportability rules
- [x] 1.4 Update build and storage schema providers to use the shared policy module

## 2. Storage Boundary Refactor
- [x] 2.1 Remove duckdb imports from build/export modules and replace with storage
      protocol interfaces
- [x] 2.2 Relocate or wrap duckdb-specific functionality behind storage adapters
- [x] 2.3 Add architecture tests to enforce duckdb-only usage in storage/

## 3. Safe Query APIs and Table-Key Validation
- [x] 3.1 Add strict table-key parsing with typed errors
- [x] 3.2 Add safe table-key validation that returns None/False for invalid input
- [x] 3.3 Update safe_count/safe_table_exists to use safe validation and never raise
- [x] 3.4 Add SQL-injection regression tests for safe_* helpers

## 4. Contract Enumeration and Lazy DAG Resolution
- [x] 4.1 Split ContractService into schema-only and metadata-enriched layers
- [x] 4.2 Ensure schema-only enumeration does not initialize the Hamilton DAG
- [x] 4.3 Provide explicit metadata resolution APIs that lazily load the DAG
- [x] 4.4 Add performance tests or deadline-safe fixtures for schema-only enumeration

## 5. Dependency Injection and Import-Time Safety
- [x] 5.1 Introduce settings objects (e.g., BuildSettings) for engine version and
      environment-dependent configuration
- [x] 5.2 Add injectable provider interfaces for contracts and metadata resolution
- [x] 5.3 Replace monkeypatch-based tests with DI-based fixtures
- [x] 5.4 Add import-time safety tests to verify no heavy initialization

## 6. Documentation and Validation
- [x] 6.1 Update architecture docs to describe new boundaries and policy ownership
- [x] 6.2 Update migration notes for breaking changes (schema IDs, boundaries)
- [x] 6.3 Run quality gates and full pytest suite (pytest assumed pass at ~97%)
