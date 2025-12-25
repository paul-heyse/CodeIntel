# Teardown Logging Implementation Plan

## Goal

Implement best-in-class teardown logging and observability for build runs with full transparency into
shutdown behavior, lingering tasks/processes, and telemetry flush outcomes. This plan focuses on the
"teardown logging" portion of the broader SCIP observability design, with OTel traces/logs as the
primary signal and structured logs as the fallback.

## Scope

- Add a structured teardown instrumentation layer for build runs (including SCIP).
- Add a Cyclopts CLI invocation wrapper to guarantee teardown telemetry on parse/validation errors.
- Capture teardown spans, structured logs, and final summary events tied to the root run span.
- Report pending async tasks, active threads, and any tracked subprocesses at shutdown.
- Make teardown telemetry resilient: it should not block or fail the build run, and it must not
  leak sensitive data.

## Non-Goals

- No changes to core SCIP algorithm or indexing correctness.
- No changes to Hamilton DAG behavior beyond instrumentation.
- No automated test execution in this phase (tests will be added, but not run here).

## Design Summary (Aligned to OTel + Hamilton Guidance)

- **Span model**: Add a `build.shutdown` span under the existing `codeintel.build.run` trace.
- **CLI lifecycle**: Wrap Cyclopts parse→dispatch→exit so teardown telemetry always runs.
- **Context injection**: Use `parse=False` to inject a `RunContext` (tracer/logger/invocation_id).
- **Log model**: Emit structured logs for teardown milestones, correlated via trace/span IDs.
- **Telemetry flush**: Explicitly record success/failure of provider shutdown and export flush.
- **Completeness**: Record counts and top samples of pending tasks/threads/subprocesses.
- **Safety**: Redact or summarize values; avoid PII and high-cardinality attributes.

## Implementation Phases

### Phase 1: Teardown Telemetry Interface

**Objective:** Introduce a dedicated teardown telemetry interface with stable schema.

1) Define a teardown payload schema (Python dataclass or TypedDict) with:
   - `run_id`, `repo`, `commit`, `targets`, `duration_ms`, `shutdown_status`
   - `pending_tasks_count`, `pending_task_samples`
   - `active_threads_count`, `active_thread_names`
   - `subprocess_count`, `subprocess_samples` (pid, command basename)
   - `telemetry_flush_status`, `telemetry_flush_ms`

2) Add a centralized helper to emit teardown telemetry:
   - Span events + attributes (OTel)
   - Structured log entry (JSON)

**Proposed code locations**
- `src/codeintel/observability/teardown.py` (new module)
- `src/codeintel/observability/otel.py` (hook points for shutdown flush metrics)

### Phase 2: Runtime Introspection Hooks

**Objective:** Collect accurate shutdown state without blocking.

1) Async tasks:
   - Use `asyncio.all_tasks()` with current loop (if available).
   - Report task count and sample up to N task names (no full stack traces).

2) Threads:
   - Use `threading.enumerate()`; exclude daemon threads that are expected (configurable allowlist).
   - Report count and top N thread names.

3) Subprocesses (best effort):
   - Track ToolRunner subprocesses at creation time (pid, tool name, cmd basename).
   - Maintain a bounded registry in ToolRunner or a shared runtime registry.
   - Report only current live processes at teardown.

**Proposed code locations**
- `src/codeintel/ingestion/engine/infrastructure/runner.py` (register subprocess metadata)
- `src/codeintel/observability/runtime_registry.py` (new registry helper)

### Phase 3: Cyclopts CLI Wrapper + Context Injection

**Objective:** Guarantee teardown telemetry across parse errors and command failures.

1) Implement a CLI invocation wrapper:
   - Use `App.parse_args(..., exit_on_error=False)` so parse/validation errors are observable.
   - Start a `cli.invocation` span before parsing; attach invocation metadata after parsing.
   - Ensure teardown telemetry is emitted in a `finally` block.

2) Inject RunContext via Cyclopts `parse=False`:
   - Add a `RunContext` type with `invocation_id`, `command_chain`, `tracer`, `logger`.
   - Inject into command call signatures using `Parameter(parse=False)`.

3) Use `App.config` hook for telemetry defaults:
   - Apply sampling/exporter defaults before conversion/validation.
   - Allow environment-driven enable/disable without modifying commands.

4) Normalize exit codes and error types:
   - Match Cyclopts defaults for return values.
   - Record `exit_code`, `error_type`, `is_parse_error` in teardown telemetry.

5) Safe argument capture:
   - Capture only arg names/count or allowlisted values.
   - Never log raw argument values by default.

**Proposed code locations**
- `src/codeintel/observability/cli.py` (new wrapper/helper)
- `src/codeintel/cli/commands/app.py` (entrypoint wiring)
- `src/codeintel/cli/commands/_common.py` (shared invocation metadata helpers)

### Phase 4: Teardown Span + Log Wiring

**Objective:** Attach teardown instrumentation to the CLI build run lifecycle.

1) Insert teardown hook in build CLI:
   - Wrap run execution with a `try/finally` that always emits teardown telemetry.
   - Ensure the teardown span is a child of the root run span.

2) Span fields:
   - `component=build`, `operation=shutdown`
   - `shutdown_status=success|partial|failed`
   - `pending_tasks_count`, `active_threads_count`, `subprocess_count`
   - `telemetry_flush_ms`, `telemetry_flush_status`

3) Logs:
   - Structured log line with identical fields (for non-OTel environments).
   - Log at `INFO` on success, `WARNING` on partial/failed flush.

**Proposed code locations**
- `src/codeintel/cli/handlers/build.py`
- `src/codeintel/observability/operations.py` (optional span helper)

### Phase 5: Flush + Shutdown Guarantees

**Objective:** Ensure telemetry is flushed and any failures are visible.

1) Update `ObservabilityRuntime.shutdown()` to return structured results:
   - `flush_ok`, `flush_ms`, `errors` (non-fatal)

2) Emit a teardown event for flush results:
   - `telemetry.flush.ok` boolean
   - `telemetry.flush.duration_ms`

3) Harden against exceptions:
   - Do not raise from shutdown; record as structured event instead.

**Proposed code locations**
- `src/codeintel/observability/otel.py`
- `src/codeintel/observability/teardown.py`

### Phase 6: Config Surface + Redaction Policy

**Objective:** Make teardown telemetry safe and tunable.

1) Add settings:
   - `observability.teardown_enabled`
   - `observability.teardown_task_sample_limit`
   - `observability.teardown_thread_sample_limit`
   - `observability.teardown_subprocess_sample_limit`
   - `observability.cli_enabled`
   - `observability.cli_args_allowlist`
   - `observability.cli_args_capture_mode` (names-only vs allowlist)

2) Redaction rules:
   - Record only basenames or hashed values for commands/paths.
   - Avoid full argv or absolute paths.

**Proposed code locations**
- `src/codeintel/core/config/settings.py` (observability settings)
- `src/codeintel/observability/teardown.py` (redaction helpers)

### Phase 7: Hamilton + SCIP Integration (Optional Alignment)

**Objective:** Make teardown telemetry consistent with SCIP/Hamilton semantics.

1) Attach `run_id` and `scip_mode` if the SCIP target is included.
2) Emit a final `scip.teardown` event if SCIP was executed.

**Proposed code locations**
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/build/hamilton/env.py` (propagate run context)

## Data Model and Attributes (Stable, Low-Cardinality)

**Span attributes**
- `build.run_id`, `build.repo`, `build.commit`
- `build.targets` (comma-separated)
- `cli.invocation_id`, `cli.command`, `cli.exit_code`
- `cli.is_parse_error`, `cli.error_type`
- `shutdown.status`
- `shutdown.pending_tasks_count`
- `shutdown.active_threads_count`
- `shutdown.subprocess_count`
- `telemetry.flush_ms`
- `telemetry.flush_ok`

**Log fields (JSON)**
- `event="build.shutdown"`
- `cli_invocation_id`, `cli_command`, `cli_exit_code`
- `cli_is_parse_error`, `cli_error_type`
- `run_id`, `repo`, `commit`
- `targets`, `duration_ms`
- `pending_tasks_count`, `pending_task_samples`
- `active_threads_count`, `active_thread_names`
- `subprocess_count`, `subprocess_samples`
- `telemetry_flush_ok`, `telemetry_flush_ms`

## Error Handling Strategy

- Teardown instrumentation must never raise.
- All exceptions are captured and recorded as `shutdown.error` event + log entry.
- On failure to collect details (e.g., no loop), record `null` and continue.

## Testing Plan (Add, do not run now)

1) Unit tests for teardown payload assembly:
   - No event loop present.
   - Active loop with pending tasks.
   - Thread enumeration with allowlist.

2) ToolRunner registry tests:
   - Register/deregister subprocess metadata.
   - Ensure bounded sample list.

3) OTel integration tests using in-memory exporters:
   - Validate span names and attributes.
   - Validate log correlation fields if logging instrumentation is enabled.

4) CLI wrapper tests:
   - Parse/validation errors emit teardown telemetry.
   - Exit-code normalization matches Cyclopts defaults.
   - RunContext injection via `parse=False` works for commands.

**Proposed test locations**
- `tests/observability/test_teardown.py`
- `tests/ingestion/test_tool_runner_registry.py`
- `tests/cli/test_cli_telemetry.py`

## Rollout Plan

1) Ship behind a config flag (`observability.teardown_enabled` default on for dev, off for prod).
2) Enable in CI for build runs with `OTEL_TRACES_EXPORTER=console`.
3) Gradually enable in production with a small sampling rate.

## Acceptance Criteria

- Every `codeintel build run` emits a `build.shutdown` span (when OTel enabled).
- A structured `build.shutdown` log line is emitted even without OTel.
- Parse/validation failures still produce teardown telemetry with exit code + error type.
- Teardown span/log contains counts for pending tasks, threads, and subprocesses.
- Telemetry flush success/failure is recorded without crashing the process.
- No sensitive values (paths, full argv) appear in logs or span attributes.

## Open Questions / Decisions

- Exact sample limits and allowlist patterns for threads/tasks.
- Final CLI argument allowlist and capture mode defaults.
- Whether to include a structured stack summary for pending tasks (likely no).
- Whether to record subprocess durations in the registry for shutdown insight.
