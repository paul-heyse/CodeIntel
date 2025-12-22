## Context
Analytics and history workflows are now DAG-first, with Hamilton targets materializing
canonical outputs. Runtime configuration parsing is centralized via the core runtime loader,
and observability settings are injected rather than parsed at call sites. Ibis connections
and row writes flow through storage-owned interfaces to reduce drift. The project direction
remains DAG-first with canonical outputs derived from Hamilton.

## Goals / Non-Goals
- Goals:
  - Make Hamilton DAG the only execution path for analytics/graph/history computation.
  - Preserve CLI/debug outputs but source them from DAG-derived datasets or cached artifacts.
  - Centralize runtime configuration loading for build/CLI/serving entrypoints, including
    observability and metrics gating settings.
  - Inject observability runtime handles from the canonical loader rather than per-surface
    bootstrap logic.
  - Enforce storage-owned Ibis access and contract-backed analytics persistence.
  - Consolidate ID normalization into canonical core utilities.
- Non-Goals:
  - Changing dataset schemas, metrics semantics, or exported formats.
  - Introducing new analytics targets or expanding contract coverage beyond current scope.

## Decisions
- Decision: Remove non-DAG orchestration entrypoints and replace them with Hamilton DAG
  targets/materializers, including CLI/debug flows.
- Decision: Use the canonical runtime loader for all entrypoints; eliminate bespoke env
  parsing and path normalization outside the loader, including observability settings and
  metrics auth gating.
- Decision: Observability bootstrap consumes injected settings/runtime handles rather than
  reading environment variables at the call site.
- Decision: Require Ibis connections to flow through storage-owned gateways and require
  analytics writes to use a shared contract-backed writer surface.
- Decision: Centralize ID normalization utilities in codeintel.core.data_models.ids.

## Alternatives considered
- Keep non-DAG paths for debugging convenience.
  - Rejected: allows drift and duplicates execution semantics.
- Allow per-surface env parsing (CLI/serving) for flexibility.
  - Rejected: creates inconsistent defaults and violates config-injection principles.

## Risks / Trade-offs
- DAG-only execution may increase runtime for CLI debug flows.
  - Mitigation: rely on cached DAG artifacts where possible and keep target granularity
    small for focused execution.
- Removing ad-hoc persistence helpers could require short-term refactors across analytics.
  - Mitigation: introduce a shared writer API and migrate incrementally within the change.

## Migration Plan
1. Remove non-DAG analytics/graph/history orchestration entrypoints and wire CLI/debug
   commands to DAG outputs.
2. Centralize persistence through contract-backed writer and storage Ibis gateway.
3. Consolidate runtime loader usage across entrypoints and delete bespoke parsing modules,
   including observability and metrics auth gating.
4. Remove duplicate ID normalization helpers.
5. Update tests/docs and validate quality gates.

## Open Questions
- None.

## Implementation Status (current)
- Complete: DAG-only execution for analytics/graph/history targets, storage-owned Ibis
  gateway usage, contract-backed writer adoption, runtime loader adoption across CLI/build/
  serving, subsystem cache refresh removal, and ID normalization consolidation.
- Remaining: docs/tests updates and quality gate completion.

## Remaining Design Detail
### Docs/tests and validation
- Update docs and tests to reflect DAG-only execution and canonical runtime loading.
- Run quality report and targeted test suites for analytics and CLI surfaces.

### Contract-backed persistence for caches
- Materialize subsystem cache tables via Hamilton targets or route refresh through the
  shared contract-backed writer rather than direct SQL.

### Docs/tests/quality gates
- Update documentation and tests to reflect DAG-only execution and canonical runtime
  loading; run the quality report and targeted suites once the above refactors land.
