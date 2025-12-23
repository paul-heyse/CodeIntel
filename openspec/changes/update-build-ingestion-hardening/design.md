## Context
Build execution and ingestion currently tolerate several failure modes:
- Build run IDs are derived from a timestamp + short suffix, which collides under
  parallel runs in the same second.
- Async ingestion nodes can leak coroutine outputs into the DAG, breaking materializers.
- SCIP ingestion accepts empty symbol/occurrence documents, masking tooling failures.
- Coverage edge construction yields mixed numeric types (Decimal vs int) and is used
  in span alignment tests that assume exact equality.
- core.repo_map inserts can violate primary keys when seed packs or tests reapply rows.
- Graph validation uses core.modules even when inventory is empty, missing the catalog
  fallback path.
- Module repositories return no summary when docs views are empty, even when core.modules
  has a row.

## Goals / Non-Goals
- Goals:
  - Deterministic, collision-resistant build run tracking.
  - Ingestion targets always return concrete results from the DAG.
  - Hard failures for empty SCIP outputs.
  - Coverage edges include all executed functions with canonical GOID types.
  - Snapshot-singleton repo_map writes with replace/upsert semantics.
  - Graph validation fallback to catalog inventory when modules are missing.
  - Safe repository fallbacks for file summaries.
- Non-Goals:
  - Redesign SCIP or coverage schemas.
  - Introduce new external dependencies or storage engines.

## Decisions
- Run IDs use full UUIDs (or canonical run ID helpers) and build run tracking is idempotent
  for duplicate run_id inserts.
- Ingestion target nodes must return concrete ExecutionResult/TargetRunRecord values; any
  async work is resolved inside the node (sync wrapper around async APIs).
- SCIP ingestion fails hard when parsed documents yield zero symbols or zero occurrences.
- Coverage edges include all executed functions, and GOID values are normalized to ints at
  read/write boundaries; alignment checks compare expected GOIDs as a subset.
- core.repo_map is a snapshot-singleton table with replace/upsert writes keyed by
  (repo, commit).
- Graph validation resolves module inventory via core.modules when present and falls back
  to catalog.module_by_path when missing.
- ModuleRepository returns a minimal file summary derived from core.modules when docs views
  yield no rows.

## Risks / Trade-offs
- Hard failures for empty SCIP outputs and coverage superset semantics are breaking changes
  for tests and downstream assumptions; mitigation is explicit test updates and logging.
- Upsert semantics can mask unintended duplicate writes; mitigation is logging and metrics
  on replacements.
- Fallback file summaries may omit analytics fields; mitigation is clearly marking missing
  fields as null and logging the fallback path.

## Migration Plan
- Add logging for run ID collisions and repo_map replacements during rollout.
- Update test harness stubs and seeds to satisfy SCIP and repo_map invariants.
- Update span alignment tests to treat coverage as a superset of expected GOIDs.
- Add validation tests for new failure semantics (SCIP empty outputs, coroutine guardrails).

## Open Questions
- None.
