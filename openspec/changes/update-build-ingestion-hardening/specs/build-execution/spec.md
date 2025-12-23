## ADDED Requirements
### Requirement: Collision-resistant build run identifiers
Build execution SHALL generate collision-resistant run identifiers and build run tracking
SHALL be idempotent for duplicate run_id inserts.

#### Scenario: Concurrent runs do not collide
- **WHEN** two build runs start within the same second
- **THEN** their run_id values are unique and both runs are recorded

#### Scenario: Duplicate run_id insert is idempotent
- **WHEN** build tracking attempts to insert a run_id that already exists
- **THEN** the operation completes without raising a constraint error

### Requirement: Ingestion targets return concrete results
Hamilton ingestion target nodes SHALL return concrete ExecutionResult or TargetRunRecord
values and SHALL NOT return coroutine objects.

#### Scenario: Async ingestion resolves inside the node
- **WHEN** an ingestion target performs async work
- **THEN** the node resolves the async work internally and returns a concrete result

### Requirement: Coverage edges include all executed functions
Coverage edge construction SHALL map executed lines to all matching functions and SHALL
normalize GOID values to canonical integers at read and write boundaries.

#### Scenario: Coverage captures multiple executed functions
- **WHEN** coverage data includes executed lines for more than one function
- **THEN** coverage edges include entries for each executed function with integer GOIDs

### Requirement: SCIP ingestion rejects empty documents
SCIP ingestion SHALL fail the target when parsed documents yield zero symbols or zero
occurrences, and it SHALL NOT persist empty scip tables.

#### Scenario: Empty SCIP output fails fast
- **WHEN** parsed SCIP documents contain zero symbols or zero occurrences
- **THEN** the SCIP target fails and no scip symbol or occurrence rows are written

### Requirement: Graph validation uses catalog fallback inventory
Graph validation checks that require module inventory SHALL use core.modules when present
and fall back to catalog.module_by_path when module inventory is empty or unavailable.

#### Scenario: Catalog fallback fills missing modules
- **WHEN** core.modules has no rows for the snapshot
- **THEN** graph validation derives module paths from the catalog module map
