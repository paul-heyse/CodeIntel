## ADDED Requirements

### Requirement: Single writer facade
All table materialization (insert, upsert, replace) SHALL be performed through a single writer
facade that enforces schema validation, column ordering, and schema hashing.

#### Scenario: Row materialization
- **WHEN** a DataSaver materializes row tuples
- **THEN** it delegates to the writer facade which resolves column order from SchemaService

#### Scenario: Ibis materialization
- **WHEN** a DataSaver materializes an Ibis table
- **THEN** it delegates to the writer facade with consistent validation and hashing

### Requirement: Adapter delegation
Hamilton DataSavers and IO adapters MUST NOT write directly to DuckDB; they SHALL call the writer
facade.

#### Scenario: DataSaver delegation
- **WHEN** a Hamilton DataSaver persists a table
- **THEN** it calls the writer facade instead of executing direct writes

### Requirement: Contract enforcement at write boundary
The writer facade SHALL enforce dataset contracts from DatasetCatalog before committing writes.

#### Scenario: Contract enforcement
- **WHEN** a write targets a table_key not present in DatasetCatalog
- **THEN** the writer facade rejects the write with a ProblemDetail error
