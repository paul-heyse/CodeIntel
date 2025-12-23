## ADDED Requirements
### Requirement: Repo map is snapshot-singleton
Storage SHALL persist at most one core.repo_map row per (repo, commit) and SHALL use
replace or upsert semantics for repo_map writes.

#### Scenario: Repo map replacement avoids conflicts
- **WHEN** a repo_map row is written for a snapshot that already has a row
- **THEN** the existing row is replaced or updated without a primary key violation

### Requirement: Repository APIs normalize GOID types
Storage repository reads SHALL normalize GOID-like identifiers to Python int values
regardless of backend numeric types.

#### Scenario: Decimal GOIDs are normalized
- **WHEN** a repository reads GOID values backed by Decimal types
- **THEN** the returned values are Python ints

### Requirement: File summary fallback for missing docs views
Module repository lookups SHALL return a minimal file summary derived from core.modules
when docs.v_file_summary yields no rows for a requested path.

#### Scenario: Summary fallback uses core modules
- **WHEN** docs.v_file_summary has no row for a module path
- **THEN** ModuleRepository returns a summary with rel_path and module data from core.modules
