## ADDED Requirements
### Requirement: Canonical SQL fingerprinting toolkit
The system SHALL centralize DuckDB SQL canonicalization and fingerprinting in storage
SQLGlot tools, and serving SHALL use the same pipeline for sql_fingerprint computation with
safe fallback hashing on parse failures.

#### Scenario: Serving uses canonical fingerprinting
- **WHEN** compiled SQL is fingerprinted for a semantic query
- **THEN** storage SQLGlot canonicalization is used and raw SQL hashing is used on parse
  failures

### Requirement: Semantic SQL diff is available for upgrade gates
Storage SQL tooling SHALL provide semantic diff output for canonicalized DuckDB SQL strings
to aid upgrade diagnostics and test failure analysis.

#### Scenario: Upgrade gate reports semantic diff
- **WHEN** canonical SQL output changes in an upgrade gate test
- **THEN** a semantic diff action list is available for diagnostics
