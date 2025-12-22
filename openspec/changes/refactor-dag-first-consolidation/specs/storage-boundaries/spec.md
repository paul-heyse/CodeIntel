## ADDED Requirements
### Requirement: Storage-owned Ibis connections only
Non-storage modules SHALL obtain Ibis connections and table expressions via storage-owned
Ibis gateways or facades, and SHALL NOT call ibis.duckdb.from_connection or construct Ibis
backends directly.

#### Scenario: Analytics uses storage Ibis gateway
- **WHEN** analytics modules query DuckDB via Ibis
- **THEN** they use the storage Ibis gateway/facade and do not call ibis.duckdb.from_connection

### Requirement: Contract-backed analytics writer is canonical
Analytics persistence outside Hamilton materializers SHALL use a shared, contract-backed
writer that validates rows via the schema registry and performs snapshot-scoped deletes,
and ad-hoc Pandera validation or direct SQL writes SHALL NOT be used.

#### Scenario: Analytics writes use the contract writer
- **WHEN** analytics rows are persisted outside Hamilton materializers
- **THEN** the shared contract-backed writer is used and no module-specific validation or
  direct SQL insert helpers are invoked
