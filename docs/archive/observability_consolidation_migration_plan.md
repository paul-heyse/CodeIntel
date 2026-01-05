"""Comprehensive observability consolidation + best-in-class redesign plan.

This plan consolidates shared functionality in src/codeintel/observability, reduces
code duplication, and upgrades the design to a best-in-class observability system.

Guiding principles
- No long-lived compatibility shims. Transitional adapters are allowed only as short-lived
  migration scaffolding and must be removed by the end of the plan.
- Consolidate before extending: unify duplicated logic first, then expand functionality
  on the consolidated surface.
- Keep semantics stable: attributes, policy constraints, and correctness must be preserved
  throughout, even as APIs are simplified or moved.
"""

# Observability Consolidation + Migration Plan

## Overview

This plan addresses the approved consolidation opportunities and design upgrades across
`src/codeintel/observability`. It is organized into phases that reflect an efficient
execution order (not strictly priority-based), with explicit decommissioning steps.

## Phase 0: Baseline + migration scaffold

Goal: capture current behavior, define target module layout, and define one-time migration
adapters (to be removed in the final phase).

Steps
1) Inventory current observability modules and responsibilities.
   - Map functions and dataclasses to conceptual layers:
     - Attribute shaping/redaction
     - Context propagation
     - Instrument cache + metrics
     - DB tracing
     - CLI telemetry
     - Runtime bootstrap + lifecycle
     - Teardown telemetry
     - Instrumentation registry and health
2) Define the target module layout (see Phase 1+ for module names).
3) Identify all external import points in the codebase (rg for `codeintel.observability.`).
4) Create a short-lived migration checklist with a list of import rewrites.
   - This list drives systematic refactor order and avoids missing imports later.

Acceptance gates
- The inventory table and import map exist in this document (append below).
- The target module layout is finalized (append below).

## Phase 1: Core semantic keys + attribute shaping consolidation

Goal: centralize attribute keys, truncation rules, and attribute shaping across all domains.

Steps
1) Introduce a `semconv_keys.py` (or similar) with canonical attribute keys.
   - Examples: `codeintel.repo`, `codeintel.commit`, `codeintel.run_id`,
     `codeintel.component`, `codeintel.operation`, db statement hash keys, etc.
2) Consolidate truncation and redaction utilities into a single module:
   - Merge truncation logic from:
     - `semconv.py`
     - `db_query_text.py`
     - `db_query_parameters.py`
     - `sql_redaction.py`
     - `teardown.py`
3) Replace local `_truncate` or bespoke trimming logic with a shared helper.
4) Replace direct attribute key string literals with `semconv_keys` constants.

Acceptance gates
- No duplicated `_truncate` helpers remain.
- Attribute keys in observability modules reference centralized constants.

## Phase 2: Policy + cardinality budget unification

Goal: unify attribute policy, allowlists, and cardinality constraints with a single source.

Steps
1) Introduce a `CardinalityBudget` or `AttributeBudget` object.
   - Include: max list length, max string length, max arg names, route/tool name truncation.
2) Move policy-driven truncation/allowlist rules to a unified helper.
3) Replace direct policy lookups in:
   - `attribute_taxonomy.py`
   - `operations.py`
   - `semconv.py`
   - `cli.py`
4) Provide a single helper to apply the budget and allowlist.

Acceptance gates
- Cardinality rules are enforced through the shared budget helper.
- All attribute shaping paths use a single function.

## Phase 3: Correlation + telemetry context refactor

Goal: unify correlation context and attribute emission into a single object.

Steps
1) Introduce `TelemetryContext`:
   - Contains repo, commit, run_id, domain, correlation_id.
   - Provides `span_attributes()` and `metric_attributes()`.
2) Replace individual getters in `db_span_emitter.py` with `TelemetryContext`.
3) Replace `CorrelationBundle` + `current_correlation_bundle()` usage in
   `operations.py` and `context.py` with `TelemetryContext`.
4) Migrate `context.py` to expose only:
   - `get_telemetry_context()`
   - `telemetry_context(...)` context manager
5) Remove legacy getters (`get_repo`, `get_commit`, etc.) by the end of Phase 8.

Acceptance gates
- All span/metric attribute emission for correlation uses the new context object.
- `context.py` exports a single modern interface only.

## Phase 4: Instrument caches + instrumentation registry consolidation

Goal: centralize metric instrument caching and registry emission.

Steps
1) Introduce a shared `InstrumentCache` utility for `WeakKeyDictionary` caching.
2) Replace per-module caches:
   - `operations.py`
   - `cli.py`
   - `instrumentation_registry.py`
3) Consolidate instrumentation status and runtime health into a single registry
   (or reuse a shared registry base class).
4) Ensure registry emits both structured logs and metrics consistently.

Acceptance gates
- All metric instruments use a single cache helper.
- Instrumentation status reporting uses a single registry path.

## Phase 5: DB tracing consolidation

Goal: simplify and standardize DB tracing configuration and attribute shaping.

Steps
1) Introduce a `DbTracingSettings` dataclass:
   - Replace the scattered db-related fields in `ObservabilityConfig` and `ObservabilityRuntime`.
2) Move db span attribute building to a single builder that is used by both:
   - `duckdb_tracing.py`
   - any future DB tracing instrumentation
3) Centralize SQL redaction policy and db.query.text handling to a single module.
4) Remove redundant query-text helpers from `db_query_text.py` when consolidated.

Acceptance gates
- All DB tracing config is grouped under `DbTracingSettings`.
- `duckdb_tracing.py` reads only from a consolidated settings object.

## Phase 6: Operation + CLI + MCP instrumentation unification

Goal: move all operational spans/metrics into a shared “operation scope”.

Steps
1) Introduce `OperationScope` (or similar) that:
   - Creates a span
   - Applies correlation attributes
   - Records duration + success metrics
2) Refactor `observe_operation` to wrap/implement this new scope.
3) Refactor CLI instrumentation in `cli.py` to use `OperationScope`.
4) Refactor MCP middleware in `mcp.py` to use `OperationScope`.
5) Ensure both span attributes and metrics are consistent across paths.

Acceptance gates
- CLI and MCP instrumentation no longer implement bespoke span/metric logic.
- All operation metrics are emitted by a single path.

## Phase 7: Observability runtime configuration restructure

Goal: reduce monolithic config + runtime objects into composable sub-configs.

Steps
1) Split `ObservabilityConfig` into sub-configs:
   - TraceSettings, MetricSettings, LogSettings, DbTracingSettings,
     InstrumentationSettings, ExporterSettings.
2) Replace direct field references in `otel.py` with sub-config references.
3) Mirror structure in `ObservabilityRuntime` for consistency.
4) Update test-mode overrides to operate on sub-config objects.

Acceptance gates
- `ObservabilityConfig` and `ObservabilityRuntime` use sub-config objects.
- No direct references to old config fields remain.

## Phase 8: Unified telemetry events and teardown pipeline

Goal: consolidate teardown payloads + event emission into a reusable abstraction.

Steps
1) Introduce a `TelemetryEvent` abstraction with:
   - `span_attributes()`
   - `event_attributes()`
   - `log_payload()`
2) Migrate `TeardownTelemetry` to extend or compose `TelemetryEvent`.
3) Move shared serialization helpers (prune, redact, coerce) into shared module.
4) Ensure shutdown and error events use the same event abstraction.

Acceptance gates
- Teardown telemetry uses the unified event abstraction.
- Shared serialization helpers are centralized.

## Phase 9: Consolidate config file loading + SDK configuration

Goal: unify config validation, file loading, and SDK configuration.

Steps
1) Create a `ConfigLoader` that:
   - Loads config files (YAML/JSON)
   - Validates structure
   - Applies to SDK using a unified entrypoint
2) Replace `config_validation.py` and `_apply_config_file` in `otel.py`
   with the `ConfigLoader`.
3) Ensure consistent logging for invalid config files.

Acceptance gates
- Only one module handles OTel config file parsing and SDK configuration.

## Phase 10: Decommission legacy code

Goal: remove shims and legacy APIs introduced during migration.

Steps
1) Remove compatibility exports and legacy entrypoints.
2) Delete unused modules after consolidation:
   - old helpers replaced by the new core modules.
3) Update all imports to the new API surface.
4) Verify no deprecated or shim paths remain.

Acceptance gates
- `rg` for removed symbols yields no results.
- No legacy compatibility functions remain in `__init__.py`.

## Phase 11: Validation + tests

Goal: ensure functionality and performance remain correct after consolidation.

Test updates
- Add/expand unit tests for:
  - attribute shaping and truncation
  - correlation context propagation
  - operation scope metrics + spans
  - DB tracing attribute emission
  - CLI telemetry behavior
  - instrumentation registry emission
  - teardown event payloads
- Focus on verifying consistent attribute keys, cardinality limits, and
  successful spans/metrics emission.

Acceptance gates
- All tests pass for observability components.
- No regressions in telemetry behavior or logs.

## Appendices

### A) Target module layout (proposed)
- `observability/semconv_keys.py` (attribute key constants)
- `observability/attribute_sanitizer.py` (truncate/redact/shape helpers)
- `observability/telemetry_context.py` (context + correlation API)
- `observability/operation_scope.py` (span + metrics lifecycle)
- `observability/instrument_cache.py` (shared instrument caching)
- `observability/db_tracing.py` (db tracing config + span builder)
- `observability/config_loader.py` (OTel config file handling)
- `observability/runtime.py` (bootstrap + runtime management)
- `observability/events.py` (TelemetryEvent abstraction)

### B) Import rewrite checklist (example)
- `codeintel.observability.attributes` → `attribute_sanitizer`
- `codeintel.observability.context` → `telemetry_context`
- `codeintel.observability.operations` → `operation_scope`
- `codeintel.observability.db_span_attributes` → `db_tracing`
- `codeintel.observability.config_validation` → `config_loader`
