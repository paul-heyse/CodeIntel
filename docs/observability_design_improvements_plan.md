# Observability Design Improvements Plan

## Goal

Deliver a best-in-class observability system with unified governance, safer defaults, and
consistent behavior across traces, metrics, logs, and DB spans. The system should be easier
to extend, harder to misuse, and simpler to test and operate.

## Design Principles

- Single source of truth for policy and attribute shaping.
- Consistent correlation and resource identity across all signals.
- Explicit lifecycle management with predictable test hooks.
- Low-cardinality defaults with opt-in expansion.
- Semconv alignment without sacrificing domain-specific metadata.

## Scope Summary

- Centralized observability policy for taxonomy, redaction, and budgets.
- Shared attribute shaping utility for all surfaces.
- Runtime manager for lifecycle and test integration.
- Pipeline health instrumentation with metrics and diagnostics.
- Semantic conventions translation layer.
- Per-surface sampling and cardinality budgets.
- Correlation bundle propagation across spans, logs, and metrics.
- Config-file validation with clear failure modes.

## Non-Goals

- No changes to business logic or DAG behavior beyond instrumentation.
- No changes to runtime features unrelated to observability.
- No automatic production rollout without explicit flags.

## Phase 0: Policy Contract and Inventory

**Objective:** Define a unified policy surface and map existing usage.

**Work items**
- Define `ObservabilityPolicy` dataclass with:
  - attribute allowlist and prefix rules
  - redaction rules for CLI args, paths, and SQL
  - cardinality budgets (per surface and per attribute)
  - coercion and truncation limits
- Inventory every attribute emitted in:
  - `src/codeintel/observability/operations.py`
  - `src/codeintel/observability/cli.py`
  - `src/codeintel/observability/db_span_emitter.py`
  - `src/codeintel/observability/teardown.py`
  - `src/codeintel/serving/http/route_utils.py`
  - `src/codeintel/observability/mcp.py`
- Document which attributes map to OTel semconv and which remain `codeintel.*`.

**Proposed code locations**
- `src/codeintel/observability/policy.py`
- `src/codeintel/observability/attribute_taxonomy.py`
- `docs/observability_policy.md` (new)

**Acceptance criteria**
- Policy object defined with explicit defaults and docstrings.
- Attribute inventory completed and mapped to policy rules.

## Phase 1: Shared Attribute Shaping Utilities

**Objective:** Consolidate all attribute shaping and avoid drift.

**Work items**
- Create a shared attribute shaping module with:
  - `coerce_attribute_value`
  - `filter_attributes`
  - `truncate_list` and `truncate_string`
  - per-surface attribute allowlists and budgets
- Replace duplicate `_coerce_attribute_value` implementations in:
  - `src/codeintel/observability/operations.py`
  - `src/codeintel/observability/db_span_emitter.py`
  - `src/codeintel/observability/teardown.py`
- Update CLI arg capture to use policy-driven budgets.

**Proposed code locations**
- `src/codeintel/observability/attributes.py`
- `src/codeintel/observability/attribute_taxonomy.py`
- `src/codeintel/observability/cli.py`

**Acceptance criteria**
- One canonical attribute shaping utility.
- All observability surfaces use shared shaping functions.

## Phase 2: Runtime Manager and Lifecycle Hardening

**Objective:** Make telemetry lifecycle explicit and test-safe.

**Work items**
- Introduce `ObservabilityRuntimeManager` that owns:
  - bootstrap
  - flush
  - shutdown
  - per-process initialization
  - test-mode behavior
- Replace direct singleton access with manager accessors in:
  - `src/codeintel/observability/otel.py`
  - `src/codeintel/observability/test_mode.py`
  - `src/codeintel/cli/commands/decorators.py`
  - `src/codeintel/observability/cli.py`
- Provide explicit hooks for pytest session boundaries.

**Proposed code locations**
- `src/codeintel/observability/runtime_manager.py`
- `tests/observability/test_runtime_manager.py`

**Acceptance criteria**
- Lifecycle calls are explicit and testable.
- No per-test shutdown blocking in xdist by default.

## Phase 3: Pipeline Health Metrics and Diagnostics

**Objective:** Make telemetry health observable and actionable.

**Work items**
- Emit metrics for:
  - export attempts
  - export failures
  - dropped spans/logs
  - queue saturation
- Extend the health checker to include:
  - last export status
  - rolling failure counters
  - exporter configuration summary
- Add structured logs for pipeline health changes.

**Proposed code locations**
- `src/codeintel/observability/otel.py`
- `src/codeintel/cli/handlers/health.py`
- `tests/cli/handlers/test_health.py`

**Acceptance criteria**
- Health check reports pipeline metrics, not just flush status.
- Operators can detect exporter failures without DEBUG logs.

## Phase 4: Semantic Conventions Mapping Layer

**Objective:** Normalize semconv usage while preserving domain metadata.

**Work items**
- Define a translation map for:
  - HTTP spans
  - DB spans
  - RPC/gRPC spans
  - CLI spans
- Apply translation map in:
  - `src/codeintel/observability/operations.py`
  - `src/codeintel/observability/db_span_attributes.py`
  - `src/codeintel/serving/http/route_utils.py`
- Keep `codeintel.*` keys as first-class domain metadata.

**Proposed code locations**
- `src/codeintel/observability/semconv.py`
- `tests/observability/test_semconv_mapping.py`

**Acceptance criteria**
- OTel semconv alignment for core surfaces.
- Domain keys retained and documented.

## Phase 5: Per-Surface Sampling and Cardinality Budgets

**Objective:** Control cost and noise without losing critical signals.

**Work items**
- Add per-surface budgets:
  - CLI arg names
  - HTTP route labels
  - MCP tool names
  - DB query parameters
- Provide per-operation overrides in config:
  - allowlist for critical paths
  - stricter limits for noisy paths
- Wire sampling rules into:
  - `src/codeintel/observability/otel.py`
  - `src/codeintel/observability/cli.py`
  - `src/codeintel/observability/operations.py`

**Proposed code locations**
- `src/codeintel/core/config/settings.py`
- `src/codeintel/observability/policy.py`
- `tests/observability/test_policy_budgets.py`

**Acceptance criteria**
- Default budgets prevent high-cardinality blow-ups.
- Overrides are explicit and validated.

## Phase 6: Correlation Bundle and Resource Identity

**Objective:** Guarantee consistent correlation across all signals.

**Work items**
- Introduce `CorrelationBundle` with:
  - `correlation_id`
  - `run_id`
  - `repo`
  - `commit`
  - `domain`
- Ensure bundle is applied to:
  - spans
  - metrics
  - logs
  - DB spans
- Add helpers to build and apply bundle consistently.

**Proposed code locations**
- `src/codeintel/observability/context.py`
- `src/codeintel/observability/operations.py`
- `src/codeintel/observability/db_span_emitter.py`
- `tests/observability/test_correlation_bundle.py`

**Acceptance criteria**
- Same correlation keys present across all signals.
- Minimal duplication of correlation logic.

## Phase 7: Config File Validation and Fail-Fast Behavior

**Objective:** Make config file behavior reliable and debuggable.

**Work items**
- Validate `OTEL_EXPERIMENTAL_CONFIG_FILE` contents:
  - schema checks
  - missing exporters
  - invalid processor settings
- Provide clear error messages with remediation hints.
- Keep config-file mode precedence explicit and documented.

**Proposed code locations**
- `src/codeintel/observability/otel.py`
- `src/codeintel/observability/config_validation.py`
- `tests/observability/test_config_validation.py`

**Acceptance criteria**
- Invalid config fails fast with structured errors.
- Config-file precedence is deterministic.

## Phase 8: Documentation and Rollout

**Objective:** Ensure safe adoption and long-term maintainability.

**Work items**
- Update docs with:
  - policy contract
  - attribute taxonomy
  - budgets and redaction defaults
  - config precedence rules
- Add a staged rollout plan:
  - dev: full telemetry with console exporters
  - staging: OTLP with conservative sampling
  - prod: enable only after SLO thresholds

**Proposed code locations**
- `docs/observability_policy.md`
- `docs/observability_best_in_class_implementation_plan.md`

**Acceptance criteria**
- Clear, self-contained documentation for operators and developers.
- Rollout checklist with explicit gating.

## Testing Plan

- Unit tests for policy, shaping, and budgets.
- Integration tests for:
  - runtime lifecycle behavior in pytest
  - pipeline health metrics
  - semconv mapping
  - config-file validation
- Regression tests for correlation bundle consistency.

## Risks and Mitigations

- Risk: Breaking instrumentation in tests.
  - Mitigation: Runtime manager with explicit test hooks and in-memory exporters.
- Risk: Cardinality guardrails drop useful attributes.
  - Mitigation: Per-surface allowlist overrides with explicit logging.
- Risk: Config file validation rejects valid edge cases.
  - Mitigation: Validation in "warn-only" mode for the first rollout.

## Deliverables Checklist

- `ObservabilityPolicy` contract and attribute taxonomy.
- Shared attribute shaping utilities.
- Runtime manager lifecycle API.
- Health pipeline metrics and diagnostics.
- Semconv translation layer.
- Per-surface sampling and budgets.
- Correlation bundle utilities.
- Config file validation and fail-fast behavior.
- Comprehensive tests and docs updates.
