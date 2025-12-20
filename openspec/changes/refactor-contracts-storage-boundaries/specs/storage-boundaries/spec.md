## ADDED Requirements
### Requirement: DuckDB isolation to storage layer
DuckDB-specific imports and types SHALL be confined to storage modules, and non-storage
modules SHALL NOT depend on duckdb symbols.

#### Scenario: Build modules do not import duckdb
- **WHEN** build and export modules are imported
- **THEN** no duckdb symbols are imported outside storage modules

### Requirement: Storage protocol interfaces
Build and export modules SHALL depend on duckdb-agnostic storage protocol interfaces for
relations and record batch readers.

#### Scenario: Export uses protocol interface
- **WHEN** build code exports a dataset
- **THEN** it interacts with storage via protocol interfaces, not duckdb types
