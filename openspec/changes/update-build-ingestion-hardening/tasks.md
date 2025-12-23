## 1. Implementation
- [ ] 1.1 Update build run ID generation to use collision-resistant IDs and make build run
  tracking idempotent for duplicate run_id inserts.
- [ ] 1.2 Ensure ingestion target nodes return concrete results (no coroutine outputs);
  add a sync wrapper for coverage_ingest and a guardrail in DAG validation.
- [ ] 1.3 Enforce SCIP artifact validation (non-empty symbols and occurrences) and update
  stub artifacts/payloads to satisfy the new contract.
- [ ] 1.4 Normalize GOID types on read/write boundaries and update coverage edge construction
  plus span alignment assertions for superset semantics.
- [ ] 1.5 Implement snapshot-singleton core.repo_map writes (replace/upsert) and update seed
  packs/helpers to use the canonical writer.
- [ ] 1.6 Centralize graph validation module inventory fallback and update checks/tests.
- [ ] 1.7 Add a module repository fallback for file summaries when docs views are empty.
- [ ] 1.8 Add or adjust tests for run ID uniqueness, SCIP failure semantics, repo_map upserts,
  and coverage alignment behavior.
