"""Observability shared components consolidation plan.

This plan extends the existing observability consolidation by introducing shared
dataclasses, registries, and lifecycle helpers that reduce duplication and improve
determinism across tracing, metrics, logs, and events.
"""

# Observability Shared Components Consolidation Plan

## Overview

This plan delivers a second consolidation pass for `src/codeintel/observability` that
standardizes shared data models, centralizes lifecycle control, and unifies attribute
schema enforcement. The sequencing is designed for efficient delivery by front-loading
foundational dataclasses and registries that later phases build upon.

## Guiding principles

- Consolidate before extending. Shared data models land first; specialized helpers follow.
- Keep runtime behavior stable while refactoring; verify with targeted tests per phase.
- Assume OpenTelemetry libraries are available in all environments.
- No long-lived compatibility shims. Temporary adapters are allowed but removed by the end.

## Scope

Deliverables in scope:
- Shared runtime, lifecycle, and event dataclasses that replace ad-hoc dictionaries.
- A single attribute schema registry with budget and cardinality policy enforcement.
- Unified telemetry context and operation descriptors used by CLI, HTTP, MCP, and DB spans.
- A shared instrumentation registry and logging pipeline.
- Consolidated DB tracing policy and SQL redaction policy.
- Tests that codify the new invariants (contract tests, golden files, property tests).

Out of scope:
- Changes to non-observability components beyond import rewrites to the new interfaces.
- Large-scale backend exporter changes beyond config normalization and provenance tracking.

## Status update (current branch)

Phase completion status
- Phase 0: Partial. Core decisions landed but the explicit call-site map + migration checklist
  are not documented in this plan.
- Phase 1: Completed. `ObservabilityRuntime`, `ObservabilityLifecycle`, and unified shutdown
  reporting are implemented and used by CLI/HTTP/MCP entry points.
- Phase 2: Mostly complete. `TelemetryContext` + `OperationDescriptor` exist and are used
  in core observability helpers; remaining work is to ensure all call sites pass descriptors
  rather than ad-hoc strings.
- Phase 3: Mostly complete. Registry + normalizer are in place; remaining work is enforcing
  "no literal keys outside the registry" with an audit/contract.
- Phase 4: Partial. `TelemetryEvent` + `emit_event` exist but are only used in teardown;
  other event-style emissions still log directly.
- Phase 5: Completed, with cleanup pending. DB tracing + logging pipeline are unified; the
  legacy DB attribute flag still exists and should be removed in cleanup.
- Phase 6: Completed. Config resolver + provenance snapshots exist and are wired into bootstrap.
- Phase 7: Completed. Telemetry contract fixture, golden-file config tests, and property-based
  normalization tests exist under `tests/observability`.

Remaining items (summary)
1) Route all event-style emissions through `TelemetryEvent`/`emit_event`.
2) Add an audit gate to enforce registry-only attribute keys across spans/logs/metrics.
3) Remove legacy/compatibility shims and flags (see cleanup plan below).

## Phase 0: Baseline map and design artifacts

Goal: finalize the shared component design and map all call sites before code changes.

Steps
1) Inventory current modules and responsibilities under `src/codeintel/observability`.
2) Build a call-site map for the following surfaces:
   - Runtime bootstrap and shutdown
   - Telemetry context getters
   - Attribute shaping and budgets
   - DB tracing helpers
   - Logging handlers and filters
3) Define the shared dataclass API surfaces and expected invariants.
4) Add a migration checklist (module-level import rewrites).

Deliverables
- Call-site map table and migration checklist.
- Finalized shared dataclass APIs (module names and attributes).

Acceptance gates
- The map covers all call sites for CLI, HTTP, MCP, and DB tracing.
- The shared APIs are documented in this plan and ready to implement.

## Phase 1: Observability runtime and lifecycle unification

Goal: introduce a single runtime object and a lifecycle controller used everywhere.

Steps
1) Define `ObservabilityRuntime` as the single owner for:
   - `ObservabilityConfig`
   - Resource attributes
   - Provider instances (tracer, meter, logger)
   - Lifecycle state and shutdown report
2) Add `ObservabilityLifecycle` with explicit phases:
   - `bootstrap()` (config resolution + provider creation)
   - `install_logging()` (handler + filter pipeline)
   - `attach_middleware()` (HTTP/MCP)
   - `shutdown()` (flush + emit final events)
3) Add `ShutdownReport` dataclass:
   - Status, flush result, duration, errors
   - Serialized for logs and span events
4) Refactor CLI, HTTP, MCP, and build bootstrap to call lifecycle methods.

Deliverables
- `ObservabilityRuntime` and `ObservabilityLifecycle` dataclasses.
- Shared shutdown report emitted once per process.

Acceptance gates
- All entry points use the lifecycle controller.
- Shutdown emits a single structured report with consistent keys.

## Phase 2: Telemetry context and operation descriptors

Goal: standardize cross-cutting correlation data and operation naming.

Steps
1) Define `TelemetryContext` with:
   - repo, commit, run_id, correlation_id, actor, component
   - `as_span_attributes()` and `as_metric_attributes()`
2) Add `OperationDescriptor` dataclass:
   - operation name, kind, component, domain, route
3) Update `OperationScope` to consume `OperationDescriptor` and `TelemetryContext`.
4) Replace ad-hoc span attribute injection in:
   - HTTP route wrappers
   - MCP middleware
   - CLI handlers
   - DB span emission

Deliverables
- `TelemetryContext` and `OperationDescriptor` used by all operation spans.

Acceptance gates
- All operation spans share a consistent naming and attribute pattern.
- Correlation fields originate only from `TelemetryContext`.

## Phase 3: Attribute schema registry and budget policy

Goal: make attribute typing and cardinality enforcement deterministic and centralized.

Steps
1) Introduce `AttributeSchema` dataclass:
   - key, type, cardinality tier, max length, redaction mode
2) Add `SemconvRegistry` as the single source of keys and schemas.
3) Implement `CardinalityBudgetPolicy` with defaults for:
   - list length, string length, argument name length
4) Create an `AttributeNormalizer`:
   - applies schema typing, truncation, redaction, and drop rules
5) Replace direct shaping logic in spans, logs, and events with the normalizer.

Deliverables
- `SemconvRegistry` and `AttributeSchema` definitions.
- Unified attribute shaping and budget enforcement.

Acceptance gates
- All attribute emission goes through `AttributeNormalizer`.
- No direct string literal keys outside the registry.

## Phase 4: Unified event emission and instrumentation registry

Goal: standardize how events are emitted and how instruments are provisioned.

Steps
1) Create a `TelemetryEvent` dataclass:
   - name, payload, span attributes, log payload, metric attributes
2) Add a single `emit_event()` path:
   - merges telemetry context
   - applies attribute normalization
   - emits to logs and span events
3) Introduce `InstrumentRegistry`:
   - lazy tracer/meter/logger access
   - shared instrument caching
4) Replace direct instrument creation in modules with registry lookups.

Deliverables
- `TelemetryEvent` dataclass and `emit_event()` helper.
- `InstrumentRegistry` and shared instrument cache.

Acceptance gates
- All event emission flows through the shared helper.
- All instrument creation uses the registry.

## Phase 5: DB tracing and logging pipeline consolidation

Goal: make DB tracing and logging consistent and policy-driven.

Steps
1) Define `DbTracingPolicy`:
   - statement mode, redaction policy, parameter handling, sampling
2) Define `SqlRedactionPolicy`:
   - drop and hash strategies for sensitive tokens and parameters
3) Implement a shared DB span builder that:
   - applies redaction policy
   - emits consistent attributes
4) Create `LoggingPipeline`:
   - handler registry, filters, correlation injector
   - shared setup for CLI and server entry points

Deliverables
- DB tracing and logging policies with a single builder and pipeline.

Acceptance gates
- DB spans and logs share consistent redaction and correlation fields.

## Phase 6: Config resolver with provenance tracking

Goal: normalize config loading and capture provenance for debugging.

Steps
1) Implement `ConfigResolver`:
   - merges defaults, env, file, CLI, and code overrides
   - captures provenance for each field
2) Add a `ResolvedConfig` snapshot:
   - serialized to logs and diagnostics
3) Update bootstrap paths to use the resolver.

Deliverables
- `ConfigResolver` and provenance-aware config snapshots.

Acceptance gates
- Config resolution is centralized and fully traceable.

## Phase 7: Tests and governance

Goal: codify the new behavior and prevent regressions.

Steps
1) Add a `TelemetryContract` fixture:
   - validates schema adherence for spans, logs, and metrics
2) Add golden-file tests for config resolution and provenance.
3) Add property-based tests for:
   - attribute normalization
   - cardinality budgets
4) Update existing observability tests to use the shared fixtures.

Deliverables
- Contract, golden-file, and property-based tests for observability.

Acceptance gates
- All new tests pass and remove the need for ad-hoc assertions.

## Sequencing summary

1) Phase 0: Baseline map and design artifacts
2) Phase 1: Runtime and lifecycle unification
3) Phase 2: Telemetry context and operation descriptors
4) Phase 3: Attribute schema registry and budgets
5) Phase 4: Event emission and instrument registry
6) Phase 5: DB tracing and logging pipeline
7) Phase 6: Config resolver with provenance
8) Phase 7: Tests and governance

## Verification checklist

- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Targeted tests per phase (observability, cli, serving, build)
- Confirm no remaining legacy imports in `src/codeintel/observability`

## Migration cleanup

- Remove all short-lived adapters and compatibility shims after Phase 7.
- Update docs and examples to reference the new shared components only.

## Legacy + compatibility cleanup plan

Goal: remove leftover shims and legacy compatibility paths so observability is single-path
and consistent with the shared components model.

Scope (current known leftovers)
- Empty package shell: `src/codeintel/cli/observability` (no active modules).
- Shim module: `src/codeintel/observability/runtime_manager.py` (re-export only).
- Legacy DB attributes flag: `emit_legacy_db_attributes` in config and DB tracing.
- Compat config loader behavior: `src/codeintel/observability/config_loader.py` probes
  multiple legacy configurator entrypoints.

Plan steps
1) Remove empty CLI observability package
   - Delete `src/codeintel/cli/observability` and its `__pycache__`.
   - Confirm no imports reference `codeintel.cli.observability`.

2) Remove `runtime_manager` shim
   - Delete `src/codeintel/observability/runtime_manager.py`.
   - Update any residual imports to use `codeintel.observability.runtime`.

3) Decommission legacy DB attributes
   - Remove `emit_legacy_db_attributes` from:
     - `src/codeintel/core/config/settings.py`
     - `src/codeintel/observability/runtime.py`
     - `src/codeintel/observability/db_tracing.py`
     - `src/codeintel/observability/duckdb_tracing.py`
   - Update tests that reference legacy attributes to assert only the canonical schema.
   - Ensure DB span output remains deterministic and policy-driven.

4) Simplify OpenTelemetry config loader (no compat probing)
   - Restrict `apply_otel_config_file` to the canonical SDK configurator entrypoint.
   - If the entrypoint is unavailable, fail fast with a clear error.
   - Update config loader tests to match the single-entrypoint behavior.

5) Registry-only attribute enforcement
   - Add a contract test that fails on non-registered attribute keys emitted by:
     - operation spans
     - DB spans
     - logs pipeline events
   - Use `TelemetryContract` to codify the invariant.

6) Document and verify cleanup
   - Update this plan's status and note removed modules/flags.
   - Run the quality report and targeted observability tests.

Deliverables
- Legacy shim modules removed.
- DB tracing emits only canonical attributes.
- Config loader uses a single supported SDK entrypoint.
- Audit test enforces registry-only keys.

Acceptance gates
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
  passes with no new warnings/errors.
- Observability test suite passes (config resolver, logs pipeline, metrics views).
- No legacy modules remain under `src/codeintel/observability` or `src/codeintel/cli/observability`.
