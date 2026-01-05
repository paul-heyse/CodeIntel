# Systemic Test Failures Remediation - Implementation Plan

## Scope and objectives

This plan addresses the systemic failures identified in the full pytest run by fixing root causes, not just individual tests. The goals are to:

- Restore test stability by eliminating cascading build failures.
- Improve runtime correctness and observability (logs, metrics, tracing).
- Align storage schemas and contracts to a single source of truth.
- Harden tool execution boundaries with explicit, validated contracts.
- Preserve the DAG-first architecture and improve extensibility.

## Non-goals

- No feature additions beyond the fixes and hardening described here.
- No changes to external tooling installation or runtime environments.
- No behavior changes outside the affected paths unless explicitly noted.

## Failure clusters (from pytest summary)

1) DuckDB connection config mismatch in threadpool execution
   - Causes missing TargetRunRecord for many targets (modules, call_graph, scip, docstrings, tests_ingest, etc.)
2) Observability bootstrap incomplete or inconsistent
   - ProxyTracerProvider missing add_span_processor
   - Log handlers and trace filters not attached
   - Metrics instruments missing
3) Schema/contract drift (core.modules missing row_hash)
4) Tool runner API mismatch (adder() missing args)
5) Testing charter violations

## Guiding design principles

- Single source of truth for config and schema definitions.
- Deterministic build outcomes: always emit a TargetRunRecord, even on failure.
- Explicit boundaries: tools, storage, observability are injected and validated.
- DAG-first behavior: target resources and execution rules should drive behavior.

## Phase 0 - Baseline capture and instrumentation

### Tasks
- Add a short diagnostic note to the plan output to capture:
  - failing tests list
  - current build settings for the harness (parallel backend, file-backed settings)
- Capture the failing test subset list for targeted re-runs.

### Acceptance criteria
- A stable short list of failing tests is recorded with cluster mapping.

### Tests
- No tests run in this phase.

## Phase 1 - DuckDB config unification (unblocks most failures)

### Problem
Threadpool adapter opens read-only connections with a different DuckDB connect config than the writer connection, which triggers DuckDB's "different configuration" error for the same file.

### Design
Create a canonical DuckDB connect configuration that is applied to all connections for a given DB file, regardless of read/write mode. The read-only connection should only differ by read_only itself, not by config options.

### Tasks
1) Introduce a canonical connect config builder in storage session code.
   - Build a single config dict for both writer and reader.
   - Ensure any read-only defaults are also applied to the writer path.
2) Remove per-connection config divergence:
   - Replace reader-only defaults with shared defaults, or
   - Ensure writer adopts the same keys with identical values.
3) Add a guardrail:
   - If a second connection is opened to the same file with a different config, raise a descriptive error before DuckDB does.
4) Ensure the threadpool adapter always reuses the canonical config when opening per-thread gateways.

### Files to update
- src/codeintel/storage/backend/duckdb_session.py
- src/codeintel/build/hamilton/adapters/parallel.py
- src/codeintel/storage/gateway/factory.py (if needed for config propagation)

### Acceptance criteria
- No "different configuration" errors occur when running threadpool-backed builds.
- Build results include TargetRunRecord entries for requested targets.

### Tests
- tests/build/hamilton/test_phase5_replication_targets.py
- tests/graphs/test_engine_nx.py
- tests/ingestion/test_harness_ingestion.py

## Phase 2 - Build results always emit TargetRunRecord (resilience)

### Problem
When a build fails early, the result lacks TargetRunRecord entries, creating opaque errors in tests and downstream logic.

### Design
Always emit a TargetRunRecord for each requested target, even when a build fails. The record should include error details and a consistent failure status.

### Tasks
1) Update the build result builder to:
   - Create a failure record for each requested target if build_error is present.
   - Attach error details and failed_targets metadata into the record.
2) Ensure the record is produced even if the executor aborts before node execution.

### Files to update
- src/codeintel/build/hamilton/executor.py
- src/codeintel/build/hamilton/run_records.py
- src/codeintel/build/hamilton/result_builder.py (if present)

### Acceptance criteria
- Missing record errors disappear; failures surface as structured TargetRunRecord failures.

### Tests
- tests/build/hamilton/test_phase5_replication_targets.py
- tests/graphs/test_callgraph_builder.py
- tests/docs_export/test_export_edge_columns.py

## Phase 3 - Observability bootstrap correctness

### Problem
The tracer provider is a proxy without add_span_processor, and log handlers/filters are not attached. Metrics instruments are missing. This indicates the bootstrap path does not initialize a real SDK provider when enabled.

### Design
Introduce a single bootstrap path that either:
- Installs real OpenTelemetry SDK providers and attaches handlers, or
- Installs a stable no-op provider with predictable behavior

Make bootstrap idempotent and explicit; do not rely on proxy providers in production/test flows.

### Tasks
1) Add an explicit ObservabilityBootstrap helper:
   - configure tracer provider
   - configure meter provider
   - attach log handler + trace correlation filter
2) Enforce a strict config precedence:
   - explicit config file overrides env defaults
   - explicit disables yield no-op providers and no handlers
3) Update CLI/test bootstraps to call the same helper.
4) Ensure logs and spans can be asserted in tests with deterministic providers.

### Files to update
- src/codeintel/observability/otel.py
- src/codeintel/observability/mcp.py (if it boots logging/trace)
- src/codeintel/observability/logs.py (if present)
- src/codeintel/cli/bootstrap.py (or equivalent)

### Acceptance criteria
- ProxyTracerProvider error is eliminated.
- Logging pipeline tests confirm handlers and filters attached.
- Metrics tests detect the expected instruments.

### Tests
- tests/observability/test_observability_smoke.py
- tests/observability/test_logs_pipeline.py
- tests/observability/test_metrics_views.py
- tests/observability/test_otel_config.py

## Phase 4 - Schema/contract alignment (core.modules row_hash)

### Problem
core.modules table is missing row_hash while contracts expect it.

### Design
Establish a single source of truth for schema definitions and derive contracts and DDL from it. Introduce a minimal migration step that adds row_hash where needed.

### Tasks
1) Identify the canonical schema source (DDL or contract catalog) and derive the other from it.
2) Add row_hash to core.modules schema and contract catalog consistently.
3) Add a schema drift guard that fails fast when contract/DDL mismatches exist.

### Files to update
- src/codeintel/storage/schema/ddl.py
- src/codeintel/storage/contracts/catalog_state.py
- src/codeintel/storage/contracts/bootstrap.py

### Acceptance criteria
- core.modules includes row_hash in DDL and contracts.
- schema alignment validation succeeds.

### Tests
- tests/storage/test_module_index.py
- tests/storage/test_docs_views.py

## Phase 5 - Tool runner API hardening

### Problem
Tool runner tests fail due to callable invocation mismatch (adder() missing args). This indicates the registry does not validate tool signatures or inputs.

### Design
Define a ToolSpec contract that includes required args, optional args, and validation. Enforce validation before execution and produce clear errors.

### Tasks
1) Introduce ToolSpec (dataclass or protocol) with:
   - name
   - required args
   - optional args
   - validation logic
2) Update tool registry to require ToolSpec objects.
3) Update ToolRunner to validate arguments before invocation and to produce actionable errors.
4) Update tests to use ToolSpec and cover validation failure modes.

### Files to update
- src/codeintel/ingestion/engine/infrastructure/runner.py (or ToolRunner module)
- src/codeintel/ingestion/engine/plugins.py (if registry lives here)
- tests/ingestion/test_runner_plumbing.py

### Acceptance criteria
- adder() error is replaced with a validation error when inputs are missing.
- Tool runner behavior is explicit and self-documenting.

### Tests
- tests/ingestion/test_runner_plumbing.py
- tests/ingestion/test_tools.py

## Phase 6 - Testing charter enforcement

### Problem
Test charter violations indicate forbidden patterns in the codebase.

### Design
Treat the charter as a first-class quality gate. Provide actionable guidance in error output and surface violations earlier.

### Tasks
1) Fix flagged patterns as indicated by the failing test output.
2) Move charter checks to pre-commit or a dedicated CI step to reduce late failures.
3) Document the rationale in the testing guide.

### Files to update
- tests/test_testing_contract.py
- docs/tests_refinement/* (if relevant)

### Acceptance criteria
- test_testing_contract passes with no violations.

## Cross-cutting updates

- Update docs to reflect the unified DuckDB config approach and observability bootstrap expectations.
- Add targeted regression tests for the config unification logic.

## Acceptance checklist (global)

- No TargetRunRecord missing for requested targets.
- Threadpool builds complete without DuckDB config errors.
- Observability tests pass with real providers or explicit no-op behavior.
- core.modules schema includes row_hash and contracts align.
- Tool runner validation produces actionable errors.
- Testing charter passes cleanly.

## Suggested test plan

Run in order to validate incrementally:

1) tests/build/hamilton/test_phase5_replication_targets.py
2) tests/graphs/test_engine_nx.py
3) tests/observability/test_observability_smoke.py
4) tests/storage/test_module_index.py
5) tests/ingestion/test_runner_plumbing.py
6) Full pytest (segmented by major directories if needed)
