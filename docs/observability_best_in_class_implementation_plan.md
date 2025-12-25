# Best-in-Class Observability Implementation Plan

## Goal

Deliver a unified, production-grade observability, telemetry, and logging system that fully
leverages OpenTelemetry (traces, metrics, logs), Hamilton telemetry, Cyclopts CLI instrumentation,
and grpcio‑observability where applicable. The outcome is a single, coherent system with strong
governance (cardinality, redaction, limits), reliable shutdown/flush semantics, and consistent
cross‑signal correlation.

## Scope Summary

- Full OTel control-plane coverage: env vars, config file, resource identity, propagators,
  exporters, batching, limits.
- Logs pipeline (SDK logs) with trace/log correlation and optional trace‑based gating.
- Trace + metric guardrails: sampling, span limits, views/exemplars.
- Cross‑signal correlation and standardized semantic conventions.
- Hamilton UI telemetry integration with safe capture policies.
- gRPC metrics integration with cardinality controls and platform gating.
- CLI telemetry hardening (parse‑time coverage, safe args).
- Instrumentation registry + health diagnostics + validation tests.

## Non‑Goals

- No changes to business logic or data model beyond instrumentation.
- No changes to existing build DAG semantics (except telemetry adapters).
- No automatic rollout to production without explicit enablement flags.

## Detailed Follow‑Up Addendum (Execution‑Ready Extensions)

This section expands the plan into concrete, execution‑ready detail: configuration maps, schemas,
ownership, and rollout/testing playbooks.

### A) Configuration Map (Env → Settings → Runtime)

Build a single, documented mapping table that captures the entire control plane:

| Source | Env/config key | Settings field | Runtime effect | Default |
| --- | --- | --- | --- | --- |
| OTel | `OTEL_SERVICE_NAME` | `observability.service_name` | `Resource.service.name` | `codeintel` |
| OTel | `OTEL_RESOURCE_ATTRIBUTES` | `observability.resource_attributes` | `Resource` merge | empty |
| OTel | `OTEL_PROPAGATORS` | `observability.propagators` | Context propagation | `tracecontext,baggage` |
| OTel | `OTEL_TRACES_SAMPLER` | `observability.traces_sampler` | Sampler selection | `parentbased_traceidratio` |
| OTel | `OTEL_TRACES_SAMPLER_ARG` | `observability.traces_sampler_arg` | Sampler config | `1.0` |
| OTel | `OTEL_EXPORTER_OTLP_ENDPOINT` | `observability.otlp_endpoint` | OTLP base endpoint | `http://localhost:4318` |
| OTel | `OTEL_EXPORTER_OTLP_PROTOCOL` | `observability.otlp_protocol` | gRPC vs HTTP | `http/protobuf` |
| OTel | `OTEL_EXPORTER_OTLP_HEADERS` | `observability.otlp_headers` | Auth headers | empty |
| OTel | `OTEL_BSP_*` | `observability.traces_batch_*` | Span batching | SDK defaults |
| OTel | `OTEL_BLRP_*` | `observability.logs_batch_*` | Log batching | SDK defaults |
| OTel | `OTEL_METRIC_EXPORT_*` | `observability.metrics_export_*` | Metric reader interval | SDK defaults |
| OTel | `OTEL_ATTRIBUTE_*` | `observability.attribute_limits` | Attribute limits | SDK defaults |
| OTel | `OTEL_SPAN_*` | `observability.span_limits` | Span/event/link limits | SDK defaults |
| OTel | `OTEL_LOGRECORD_*` | `observability.log_limits` | Log record limits | SDK defaults |
| OTel | `OTEL_EXPERIMENTAL_CONFIG_FILE` | `observability.config_file` | Full SDK config | unset |
| Python | `OTEL_PYTHON_LOG_CORRELATION` | `observability.log_correlation` | Inject trace IDs | false |
| Python | `OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED` | `observability.logs_auto_instrument` | Attach OTLP handler | false |
| Hamilton | `HAMILTON_*` | `observability.hamilton_capture_*` | UI capture policy | defaults |
| gRPC | `GRPC_PYTHON_CENSUS_*` | `observability.grpc_batch_*` | gRPC metrics buffer | defaults |

Notes:
- Build a single parser with strict validation (warn + ignore invalid values).
- Ensure config file mode overrides env vars except env‑substitution inside the file.

#### Current defaults from runtime settings (CodeIntel today)

These defaults are derived from `src/codeintel/core/runtime/loader.py` and
`src/codeintel/core/config/settings.py` as of now.

| Env var | Settings field | Default | Notes |
| --- | --- | --- | --- |
| `OTEL_SDK_DISABLED` | `observability.enabled` | `false` | Enabled when unset. |
| `OTEL_SERVICE_NAME` | `observability.service_name` | `None` | Unset means SDK default. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `observability.otlp_endpoint` | `None` | Unset means SDK default. |
| `CODEINTEL_EXPORT_TRACES` | `observability.export_traces` | `true` | Trace exporter on. |
| `CODEINTEL_EXPORT_METRICS` | `observability.export_metrics` | `true` | Metrics exporter on. |
| `CODEINTEL_CONSOLE_TELEMETRY` | `observability.console_export` | `false` | Console exporter off. |
| `CODEINTEL_PROMETHEUS_METRICS` | `observability.prometheus_enabled` | `false` | Prometheus reader off. |
| `CODEINTEL_OBSERVABILITY_TEARDOWN_ENABLED` | `observability.teardown_enabled` | `true` | Teardown telemetry on. |
| `CODEINTEL_OBSERVABILITY_TEARDOWN_TASK_SAMPLE_LIMIT` | `observability.teardown_task_sample_limit` | `5` | Pending task sample size. |
| `CODEINTEL_OBSERVABILITY_TEARDOWN_THREAD_SAMPLE_LIMIT` | `observability.teardown_thread_sample_limit` | `5` | Thread sample size. |
| `CODEINTEL_OBSERVABILITY_TEARDOWN_SUBPROCESS_SAMPLE_LIMIT` | `observability.teardown_subprocess_sample_limit` | `5` | Subprocess sample size. |
| `CODEINTEL_OBSERVABILITY_CLI_ENABLED` | `observability.cli_enabled` | `true` | CLI telemetry on. |
| `CODEINTEL_OBSERVABILITY_CLI_ARGS_ALLOWLIST` | `observability.cli_args_allowlist` | empty | No allowlisted args. |
| `CODEINTEL_OBSERVABILITY_CLI_ARGS_CAPTURE_MODE` | `observability.cli_args_capture_mode` | `names-only` | Names-only capture. |
| `CODEINTEL_OTEL_DUCKDB_TRACING` | `observability.duckdb_tracing_enabled` | `true` | DuckDB spans on. |
| `CODEINTEL_OTEL_DUCKDB_REQUIRE_PARENT` | `observability.duckdb_require_parent_span` | `true` | Require parent span. |
| `CODEINTEL_OTEL_DB_STATEMENT_MODE` | `observability.duckdb_statement_mode` | `hash` | Allowed: full/hash/operation/none. |
| `CODEINTEL_OTEL_DB_STATEMENT_HASH_LEN` | `observability.duckdb_statement_hash_len` | `16` | Hash length. |
| `CODEINTEL_OTEL_DB_QUERY_SUMMARY_MAX_LEN` | `observability.duckdb_query_summary_max_len` | `255` | Summary length cap. |
| `CODEINTEL_OTEL_DB_QUERY_SUMMARY_MAX_TARGETS` | `observability.duckdb_query_summary_max_targets` | `6` | Target cap. |
| `CODEINTEL_OTEL_DB_QUERY_SUMMARY_EMIT_ELLIPSIS` | `observability.duckdb_query_summary_emit_ellipsis` | `true` | Ellipsis on truncation. |
| `CODEINTEL_OTEL_DB_QUERY_SUMMARY_HASH_SUSPICIOUS` | `observability.duckdb_query_summary_hash_suspicious_targets` | `true` | Hash suspicious targets. |
| `CODEINTEL_OTEL_DB_QUERY_SUMMARY_HASH_LEN` | `observability.duckdb_query_summary_hash_len` | `12` | Summary hash length. |
| `CODEINTEL_OTEL_DB_QUERY_SUMMARY_HASH_MIN_LEN` | `observability.duckdb_query_summary_hash_min_len` | `64` | Minimum length to hash. |
| `CODEINTEL_OTEL_DB_QUERY_SUMMARY_INCLUDE_SUBQUERY_OPS` | `observability.duckdb_query_summary_include_subquery_operations` | `true` | Include subquery ops. |
| `CODEINTEL_OTEL_DB_QUERY_SUMMARY_INCLUDE_MULTI_STATEMENT` | `observability.duckdb_query_summary_include_multi_statement` | `true` | Include multi-statement. |
| `CODEINTEL_OTEL_DB_QUERY_SUMMARY_SPAN_NAME_HOOK` | `observability.db_query_summary_span_name_hook` | `false` | Span name hook off. |
| `CODEINTEL_OTEL_DB_QUERY_TEXT_POLICY` | `observability.duckdb_query_text_policy` | `never` | Allowed: never/parameterized/redacted/parameterized_or_redacted/full. |
| `CODEINTEL_OTEL_DB_QUERY_TEXT_MAX_LEN` | `observability.duckdb_query_text_max_len` | `4096` | Query text cap. |
| `CODEINTEL_OTEL_DB_QUERY_TEXT_STRIP_COMMENTS` | `observability.duckdb_query_text_strip_comments` | `true` | Comment stripping on. |
| `CODEINTEL_OTEL_DB_QUERY_TEXT_COLLAPSE_IN_LISTS` | `observability.duckdb_query_text_collapse_in_lists` | `true` | In-list collapse on. |
| `CODEINTEL_OTEL_DB_QUERY_PARAMETER_ENABLED` | `observability.duckdb_query_parameter_enabled` | `false` | Parameter capture off. |
| `CODEINTEL_OTEL_DB_QUERY_PARAMETER_KEYS` | `observability.duckdb_query_parameter_keys` | empty | No keys captured. |
| `CODEINTEL_OTEL_DB_QUERY_PARAMETER_HASH_KEYS` | `observability.duckdb_query_parameter_hash_keys` | empty | No hash keys. |
| `CODEINTEL_OTEL_DB_QUERY_PARAMETER_REQUIRE_IN_SQL` | `observability.duckdb_query_parameter_require_in_sql` | `true` | Require param in SQL. |
| `CODEINTEL_OTEL_DB_QUERY_PARAMETER_MAX_STRLEN` | `observability.duckdb_query_parameter_max_str_len` | `80` | Param string cap. |

#### SDK default reference (OpenTelemetry Python)

These defaults come from the OpenTelemetry Python SDK. They apply when CodeIntel does not
explicitly override a setting via code or env.

| SDK surface | Env vars | Default | Notes |
| --- | --- | --- | --- |
| Traces sampler | `OTEL_TRACES_SAMPLER` | `parentbased_traceidratio` | Head sampler baseline. |
| Traces sampler arg | `OTEL_TRACES_SAMPLER_ARG` | `1.0` | Sample all by default. |
| BatchSpanProcessor schedule delay | `OTEL_BSP_SCHEDULE_DELAY` | `5000` ms | Span export interval. |
| BatchSpanProcessor max queue | `OTEL_BSP_MAX_QUEUE_SIZE` | `2048` | Span buffer capacity. |
| BatchSpanProcessor max batch | `OTEL_BSP_MAX_EXPORT_BATCH_SIZE` | `512` | Export batch size. |
| BatchSpanProcessor export timeout | `OTEL_BSP_EXPORT_TIMEOUT` | `30000` ms | Export timeout. |
| Periodic metric export interval | `OTEL_METRIC_EXPORT_INTERVAL` | `60000` ms | Default metric interval. |
| Periodic metric export timeout | `OTEL_METRIC_EXPORT_TIMEOUT` | `30000` ms | Metric export timeout. |
| LogRecord processor schedule delay | `OTEL_BLRP_SCHEDULE_DELAY` | `5000` ms | Log export interval. |
| LogRecord processor max queue | `OTEL_BLRP_MAX_QUEUE_SIZE` | `2048` | Log buffer capacity. |
| LogRecord processor max batch | `OTEL_BLRP_MAX_EXPORT_BATCH_SIZE` | `512` | Log export batch. |
| LogRecord processor export timeout | `OTEL_BLRP_EXPORT_TIMEOUT` | `30000` ms | Log export timeout. |
| Attribute value length limit | `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT` | SDK default | Enforced via SpanLimits. |
| Attribute count limit | `OTEL_ATTRIBUTE_COUNT_LIMIT` | SDK default | Per-span/log attributes. |
| Span event count limit | `OTEL_SPAN_EVENT_COUNT_LIMIT` | SDK default | Per-span events. |
| Span link count limit | `OTEL_SPAN_LINK_COUNT_LIMIT` | SDK default | Per-span links. |
| Event attribute count limit | `OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT` | SDK default | Per-event attrs. |
| Link attribute count limit | `OTEL_LINK_ATTRIBUTE_COUNT_LIMIT` | SDK default | Per-link attrs. |
| LogRecord attribute count limit | `OTEL_LOGRECORD_ATTRIBUTE_COUNT_LIMIT` | SDK default | Per-log attrs. |
| LogRecord value length limit | `OTEL_LOGRECORD_ATTRIBUTE_VALUE_LENGTH_LIMIT` | SDK default | Per-log values. |

Notes:
- The Python SDK resolves these via `SpanLimits` and processor defaults. Use this table to
  confirm behavior when env vars are not set.
- When `OTEL_EXPERIMENTAL_CONFIG_FILE` is used, SDK defaults come from the config file instead.

### K) Collector Config Blueprint

Add an explicit collector pipeline blueprint that matches CodeIntel’s defaults:

1) **Receivers**
   - `otlp` receiver with gRPC (4317) and HTTP (4318).

2) **Processors**
   - `batch` with explicit queue sizes/timeouts aligned to SDK defaults.
   - `memory_limiter` to avoid process OOM during bursts.
   - `attributes` for redaction / allowlist of `codeintel.*` keys.
   - `resource` to inject `deployment.environment.name` if missing.
   - `tail_sampling` (optional in prod) with:
     - error‑based sampling
     - latency‑based sampling
     - route‑based sampling for high‑value operations

3) **Exporters**
   - OTLP exporters to vendor backends.
   - `logging` exporter (dev only).

4) **Connectors**
   - `spanmetrics` connector (optional) to generate RED metrics from traces.

Provide a reference collector YAML file in `docs/observability/collector_reference.yaml`.

### L) Attribute Taxonomy + Schemas

Define and document the exact attribute schema for each surface:

- **CLI**
  - Required: `cli.invocation_id`, `cli.command`, `cli.exit_code`, `cli.duration_ms`.
  - Optional: `cli.error_type`, `cli.is_parse_error`, `cli.arg_count`, `cli.arg_names`.

- **Build**
  - Required: `build.run_id`, `build.repo`, `build.commit`, `build.targets`.
  - Optional: `build.duration_ms`, `build.status`, `build.error`.

- **Storage / DB**
  - Required: `db.system`, `db.operation`, `db.query.summary`.
  - Optional: `db.query.hash`, `db.statement` (redacted or hashed).

- **Hamilton**
  - Required: `dag.name`, `dag.version`, `dag.project_id`, `run.id`.
  - Optional: `trace.id`, `run.tags`.

- **gRPC**
  - Required: `grpc.method`, `grpc.status`.
  - Optional: `grpc.target` (filtered), `grpc.lb.*`.

Also define max cardinality rules (e.g., allowlist for CLI args and method names).

### M) Sampling + Cost Model

Create an explicit sampler policy matrix:

| Environment | Traces sampler | Rate | Logs gating | Notes |
| --- | --- | --- | --- | --- |
| Dev | `parentbased_traceidratio` | `1.0` | off | Full fidelity. |
| Staging | `parentbased_traceidratio` | `0.2` | on | Validate costs. |
| Prod | `parentbased_traceidratio` | `0.05` | on | Tail sampling in collector. |

Include guidance for:
- “always keep errors”
- “keep slow operations”
- “raise sampling for targeted troubleshooting windows”

### N) Metric View Specifications

Define explicit metric view settings for:

- `codeintel.operation.duration_ms` histogram buckets.
- HTTP latency histograms (aligned with HTTP semconv buckets).
- gRPC call duration histograms (aligned with gRPC semconv).

Provide a reference table with buckets and rationale.

### O) Operational Runbooks

Add a runbook section:

1) **Telemetry missing**
   - Check `OTEL_*` env vars.
   - Confirm collector endpoint reachability.
   - Confirm exporter errors in logs.

2) **High drop rate**
   - Increase BSP/BLRP queue size or reduce sampling.
   - Enable tail‑sampling at collector.

3) **Unexpected cardinality**
   - Review attribute allowlists.
   - Verify argument redaction.

### P) Version Pin + Compatibility Matrix

Document supported versions and constraints:

- OTel SDK version pin.
- `grpcio-observability` Linux‑only note and tested version.
- Hamilton UI tracker version compatibility.

Include a “known gaps” list for grpc.aio metrics and logs pipeline maturity.

### B) Ownership + Lifecycle Graph

Define clear module ownership and init/shutdown ordering:

1) `observability/otel.py`: build providers (traces/metrics/logs), configure propagators.
2) `observability/operations.py`: expose span + metric helpers.
3) `observability/cli.py`: invocation wrapper + run context injection.
4) `observability/teardown.py`: teardown event emission and flush results.
5) `observability/grpc.py` (new): gRPC metrics plugin lifecycle.
6) Hamilton integration: tracker adapter attached in driver construction.

Shutdown order: CLI teardown → flush observability → shutdown providers.

### C) Telemetry Data Contracts (Per Surface)

Define schemas for each surface with required/optional fields and cardinality policy:

- **Build teardown log**: `event`, `run_id`, `repo`, `commit`, `targets`, `duration_ms`,
  `shutdown_status`, `telemetry_flush_ok`, `telemetry_flush_ms`.
- **CLI invocation span/log**: `cli.invocation_id`, `cli.command`, `cli.exit_code`,
  `cli.is_parse_error`, `cli.error_type`, `cli.duration_ms`.
- **Hamilton run**: `dag_name`, `project_id`, `environment`, `version`, `run_id`, `trace_id`.
- **gRPC metrics**: `grpc.method`, `grpc.target`, `grpc.status` with filters to cap cardinality.

### D) Collector Topology Assumptions

Document the expected collector pipeline:

- OTLP receiver (gRPC/HTTP).
- Batch processor + tail sampling + redaction/attributes processors.
- Exporters to target backend(s).
- Optional spanmetrics connector for RED metrics.

### E) Backpressure + Limits Policy

Define explicit queue sizes/timeouts/batch sizes per signal:

- Traces: `OTEL_BSP_*` defaults + rationale.
- Logs: `OTEL_BLRP_*` defaults + rationale.
- Metrics: export interval/timeout.
- Span/event/link limits to cap cardinality.

### F) Instrumentation Coverage Matrix

Provide a matrix:

| Library | Instrumentation | Mode | Status |
| --- | --- | --- | --- |
| httpx | `opentelemetry-instrumentation-httpx` | auto | enabled |
| requests | `opentelemetry-instrumentation-requests` | auto | enabled |
| asyncio | `opentelemetry-instrumentation-asyncio` | auto | enabled |
| logging | `opentelemetry-instrumentation-logging` | auto | enabled |
| grpc | `grpcio-observability` | manual | gated |

### G) Rollout Playbook

1) Dev: enable full signals + console exporters.
2) Staging: enable OTLP exporters + sampling + limits.
3) Prod: enable with conservative sampling, logs gated by trace sampling.
4) Rollback: disable via config flags + disable OTLP exporter.

### H) Testing Blueprint

Add test suites per phase:

- Config parsing: env vars, config file precedence.
- Logs pipeline: log correlation fields + OTLP export.
- Metrics views: bucket/aggregation checks.
- CLI parse errors: telemetry flush + correlation.
- gRPC plugin: registration + cardinality filter behavior.

### I) Operational Dashboards + Alerts

Define dashboards and alerts for:

- Export failure rate
- Queue saturation/drop count
- Sampling effectiveness
- Telemetry flush failures

### J) Security Review Checklist

- Redaction rules for CLI args, file paths, secrets.
- Cardinality budgets and attribute allowlists.
- Ensure no raw stack traces in logs by default.

## Phase 0: Baseline Inventory + Control-Plane Design

**Objective:** Establish a single, authoritative configuration model and inventory of current
instrumentation.

1) **Inventory and mapping**
   - Enumerate current instrumentation in `src/codeintel/observability/otel.py` and
     `src/codeintel/observability/operations.py`.
   - Identify which OTel capabilities are currently missing (logs pipeline, full env/config
     parsing, samplers, limits, views).

2) **Control‑plane specification**
   - Define a unified `ObservabilityConfig` schema in `src/codeintel/observability/otel.py`
     that includes:
     - Resource attributes (service name/version/environment)
     - Propagators (`OTEL_PROPAGATORS`)
     - Sampler (`OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG`)
     - Exporters (OTLP endpoint, protocol, headers, compression, TLS, timeout)
     - Processor/reader tuning (BSP/BLRP/metric reader intervals)
     - Limits (`OTEL_ATTRIBUTE_*`, `OTEL_SPAN_*`, `OTEL_EVENT_*`, `OTEL_LINK_*`,
       `OTEL_LOGRECORD_*`)
     - Logs pipeline options (enabled, trace‑based filter)
   - Define the precedence rules:
     - `OTEL_EXPERIMENTAL_CONFIG_FILE` overrides all SDK env‑vars except env‑substitution.
     - If no config file, apply env vars; allow explicit code overrides for
       `service.name` and `service.version` when desired.

**Acceptance criteria**
- A written configuration contract (doc + code comments) that covers all required env vars
  and precedence rules.
- A config object that can be constructed exclusively from env vars, or from a config file.

## Phase 1: Core OTel Bootstrap + Resource Identity

**Objective:** Build a best‑practice OTel bootstrap pipeline for traces and metrics.

1) **Resource identity**
   - Construct `Resource.create` with `service.name`, `service.version`,
     `deployment.environment.name`, `codeintel.repo`, and `codeintel.commit`.
   - Ensure `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES` are honored unless explicitly
     overridden by code.

2) **Sampler + span limits**
   - Implement sampler selection (parent‑based, traceid ratio, etc.) via env vars.
   - Thread `SpanLimits` into `TracerProvider` to honor attribute/event/link limits.

3) **Exporter + batching**
   - Support OTLP protocol selection (`grpc` vs `http/protobuf`).
   - Support per‑signal endpoints and headers.
   - Implement BSP knobs (`OTEL_BSP_*`) and metric reader knobs
     (`OTEL_METRIC_EXPORT_INTERVAL`, `OTEL_METRIC_EXPORT_TIMEOUT`).

4) **Propagators**
   - Configure W3C Trace Context + Baggage by default, configurable via `OTEL_PROPAGATORS`.

**Proposed code locations**
- `src/codeintel/observability/otel.py`
- `src/codeintel/core/runtime/loader.py` (env var parsing)

**Acceptance criteria**
- A single bootstrap path that respects env/config file precedence.
- Trace/metric exporters fully configurable from env vars.
- Explicit resource attributes in every span/metric.

## Phase 2: Logs Pipeline + Correlation

**Objective:** Add a first‑class logs pipeline with correlation and safe defaults.

1) **SDK logs pipeline**
   - Add `LoggerProvider` + `BatchLogRecordProcessor` + OTLP log exporter.
   - Honor `OTEL_LOGS_EXPORTER`, `OTEL_BLRP_*`, and OTLP log endpoint/protocol config.

2) **Trace/log correlation**
   - Enable trace/span injection into stdlib logs (via
     `OTEL_PYTHON_LOG_CORRELATION=true` or custom record factory).
   - Standardize log record fields for trace/span identifiers.

3) **Trace‑based log gating**
   - Optional filter to drop logs when trace is unsampled (configurable).

**Proposed code locations**
- `src/codeintel/observability/otel.py`
- `src/codeintel/observability/context.py` (correlation helpers)

**Acceptance criteria**
- Logs flow through OTLP exporter when enabled.
- Logs include trace/span IDs when a span is active.
- Log sampling respects trace sampling when configured.

## Phase 3: Metrics Views + Exemplars

**Objective:** Introduce best‑practice views and trace exemplars for metrics.

1) **Views for core metrics**
   - Provide configurable histogram buckets for latency metrics.
   - Normalize names/attributes to semconv recommendations.

2) **Exemplar filter**
   - Add `OTEL_METRICS_EXEMPLAR_FILTER` support with trace‑based option.

**Proposed code locations**
- `src/codeintel/observability/otel.py`
- `src/codeintel/observability/operations.py`

**Acceptance criteria**
- Metrics use stable naming/buckets and can be customized via config.
- Exemplars appear when traces are active (when supported).

## Phase 4: Cross‑Signal Correlation + Semantic Conventions

**Objective:** Normalize attributes and ensure consistent correlation keys.

1) **Common attributes**
   - Ensure `codeintel.run_id`, `codeintel.component`, and `codeintel.operation` are
     consistently applied to spans, metrics, and logs.
   - Include `cli.invocation_id` for CLI‑driven runs.

2) **Semantic conventions**
   - Align HTTP/DB/span attributes to OTel semconv.
   - Keep `codeintel.*` for domain‑specific metadata.

**Proposed code locations**
- `src/codeintel/observability/operations.py`
- `src/codeintel/observability/teardown.py`
- `src/codeintel/observability/db_span_attributes.py`

**Acceptance criteria**
- Consistent attribute taxonomy across all signals.
- No high‑cardinality attributes in default paths.

## Phase 5: Hamilton UI Telemetry Integration

**Objective:** Integrate Hamilton’s UI tracker with safe capture policies.

1) **Tracker adapter**
   - Attach `HamiltonTracker` or `AsyncHamiltonTracker` via
     `Builder.with_adapters(...)` when enabled.
   - Tags include environment, team, and semantic DAG version.

2) **Safe capture policy**
   - Set `HAMILTON_CAPTURE_DATA_STATISTICS` default to False in prod.
   - Reduce list/dict capture sizes by default, with override in dev.

3) **Trace linkage**
   - Include trace/run identifiers in tracker metadata.

**Proposed code locations**
- `src/codeintel/build/hamilton/driver_factory.py`
- `src/codeintel/build/hamilton/observability.py`
- `src/codeintel/core/runtime/loader.py` (settings surface)

**Acceptance criteria**
- Hamilton UI receives DAG version + run telemetry on enabled builds.
- Capture policies prevent sensitive data leakage by default.

## Phase 6: gRPC Observability

**Objective:** Add grpcio‑observability metrics with cardinality guards.

1) **Plugin registration**
   - Initialize `OpenTelemetryPlugin` with `meter_provider`.
   - Gate on Linux‑only support and `grpcio-observability` availability.

2) **Cardinality filters**
   - Use `generic_method_attribute_filter` to prevent `grpc.method` blow‑up.
   - Keep per‑method metrics only for registered endpoints.

**Proposed code locations**
- `src/codeintel/observability/grpc.py` (new)
- `src/codeintel/serving/grpc` (integration point)

**Acceptance criteria**
- gRPC metrics present when plugin is enabled.
- Unregistered method names are bounded to `grpc.method="other"` unless allowed.

## Phase 7: CLI Telemetry Enhancements

**Objective:** Expand CLI telemetry coverage and safety.

1) **Parse‑time metrics + logging**
   - Emit metrics for parse errors and parse durations.
   - Include exit code, error type, and invocation ID.

2) **Safe argument capture**
   - Centralize allowlist and names‑only capture policies.
   - Ensure arguments never appear raw by default.

**Proposed code locations**
- `src/codeintel/observability/cli.py`
- `src/codeintel/cli/commands/_common.py`

**Acceptance criteria**
- Parse failures emit telemetry with correct error classification.
- Arguments are always safe‑captured.

## Phase 8: Instrumentation Registry + Diagnostics

**Objective:** Make instrumentation coverage explicit and testable.

1) **Registry**
   - Track which instrumentations were enabled, unavailable, or suppressed.
   - Emit startup summary logs and a diagnostic metric.

2) **Health check**
   - Add a “telemetry pipeline check” command that emits a known span/log/metric
     and validates exporter configuration.

**Proposed code locations**
- `src/codeintel/observability/instrumentation_registry.py` (new)
- `src/codeintel/cli/commands/health.py`

**Acceptance criteria**
- Operators can see instrumentation coverage at startup.
- Health check verifies end‑to‑end path to Collector.

## Phase 9: Testing + Rollout

**Objective:** Validate correctness and enable staged rollout.

1) **Unit tests**
   - Config parsing (env vars + config file precedence).
   - Span limits, sampler selection, exporter options.
   - Log correlation and trace‑based log gating.
   - Metric views and exemplar filter.

2) **Integration tests**
   - In‑memory exporters for traces/metrics/logs verifying attributes.
   - CLI parse‑error telemetry emission.

3) **Rollout plan**
   - Ship disabled for prod; enable via config in dev/staging.
   - Provide a documented default config (Collector endpoint + sampling).

**Proposed test locations**
- `tests/observability/test_otel_config.py`
- `tests/observability/test_logs_pipeline.py`
- `tests/observability/test_metrics_views.py`
- `tests/cli/test_cli_telemetry.py`

**Acceptance criteria**
- CI verifies telemetry pipelines with in‑memory exporters.
- Production rollout can be toggled with config flags.

## Open Questions / Decisions

- Whether to make logs pipeline default‑on or dev‑only.
- Default sampler rate for dev vs prod.
- Whether to add support for `OTEL_EXPERIMENTAL_RESOURCE_DETECTORS`.
- Whether to expose trace‑based log gating by default.

## Deliverables Checklist

- Unified `ObservabilityConfig` and loader.
- Trace + metrics pipeline aligned to env/config.
- Logs pipeline with correlation + gating.
- Hamilton tracker integration and safe capture defaults.
- gRPC metrics plugin integration.
- CLI telemetry enhancements.
- Instrumentation registry + health check.
- Comprehensive tests + documentation updates.
