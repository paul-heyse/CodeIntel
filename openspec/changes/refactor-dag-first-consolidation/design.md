## Context
Analytics and history workflows still include non-DAG orchestration and persistence paths.
Runtime configuration parsing is duplicated across CLI/build/serving, and observability
bootstrap pulls configuration directly from the environment in multiple places. Ibis
connections and row writes also vary by module, which weakens the storage boundary and makes
drift more likely. The project direction is DAG-first with canonical outputs derived from
Hamilton.

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
- Complete: storage-owned Ibis gateway usage, contract-backed writer introduced and adopted
  across most analytics persistence, and ID normalization consolidation.
- Partial: runtime loader adoption (serving/CLI observability is centralized, but CLI config
  env parsing and serving_factory still read env directly).
- Remaining: DAG-only execution and public API cleanup, subsystem cache refreshes still
  bypass contract writer, docs/tests updates, and quality gate completion.

## Remaining Design Detail
### DAG-only execution and public API cleanup
- Hamilton targets should consume row-producing functions only and materialize via
  DataSavers; remove compute_* orchestration wrappers and any non-DAG module-level
  orchestration entrypoints.
- Remove non-DAG exports from analytics public APIs (`analytics.graphs`, `analytics.testing`,
  `analytics.subsystems`, `analytics.semantic_roles`) so only pure compute utilities remain.
- CLI/debug flows should read DAG-produced tables or cached artifacts, triggering DAG targets
  when required rather than invoking compute_* helpers directly.

### Canonical runtime loader for CLI/serving
- Replace serving_factory env parsing with runtime loader settings, and collapse CLI config
  env parsing into the canonical loader to avoid per-surface defaults.

### Contract-backed persistence for caches
- Materialize subsystem cache tables via Hamilton targets or route refresh through the
  shared contract-backed writer rather than direct SQL.

### Docs/tests/quality gates
- Update documentation and tests to reflect DAG-only execution and canonical runtime
  loading; run the quality report and targeted suites once the above refactors land.
