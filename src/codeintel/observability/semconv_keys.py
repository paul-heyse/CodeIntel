"""Semantic attribute key constants for observability."""

from __future__ import annotations

CODEINTEL_COMPONENT = "codeintel.component"
CODEINTEL_OPERATION = "codeintel.operation"
CODEINTEL_SUCCESS = "codeintel.success"
CODEINTEL_ENDPOINT = "codeintel.endpoint"
CODEINTEL_OUTPUT_FORMAT = "codeintel.output_format"
CODEINTEL_HEALTH_CHECK = "codeintel.health_check"

CODEINTEL_CORRELATION_ID = "codeintel.correlation_id"
CODEINTEL_RUN_ID = "codeintel.run_id"
CODEINTEL_DOMAIN = "codeintel.domain"
CODEINTEL_REPO = "codeintel.repo"
CODEINTEL_COMMIT = "codeintel.commit"
CODEINTEL_ACTOR = "codeintel.actor"
CODEINTEL_STORAGE_READ_ONLY = "codeintel.storage.read_only"

CODEINTEL_QUERY_ENDPOINT = "codeintel.query.endpoint"
CODEINTEL_QUERY_ROW_COUNT = "codeintel.query.row_count"
CODEINTEL_QUERY_TRUNCATED = "codeintel.query.truncated"
CODEINTEL_QUERY_VIEW_ID = "codeintel.query.view_id"
CODEINTEL_QUERY_HASH = "codeintel.query.hash"
CODEINTEL_QUERY_SCHEMA_HASH = "codeintel.query.schema_hash"

HTTP_METHOD = "http.method"
HTTP_ROUTE = "http.route"

MCP_METHOD = "mcp.method"
MCP_TOOL_NAME = "mcp.tool_name"

DB_SYSTEM_NAME = "db.system.name"
DB_NAMESPACE = "db.namespace"
DB_STATEMENT = "db.statement"
DB_QUERY_SUMMARY = "db.query.summary"
DB_QUERY_TEXT = "db.query.text"
DB_QUERY_PARAMETER_PREFIX = "db.query.parameter."

CODEINTEL_DB_STATEMENT_SHA256 = "codeintel.db.statement.sha256"

BUILD_RUN_ID = "build.run_id"
BUILD_REPO = "build.repo"
BUILD_COMMIT = "build.commit"
BUILD_TARGETS = "build.targets"
BUILD_DURATION_MS = "build.duration_ms"
BUILD_DECISION_TRACE_ARTIFACT = "build.decision_trace_artifact"
BUILD_VALIDATION_MODE = "build.validation_mode"
BUILD_VALIDATION_ISSUE_COUNT = "build.validation_issue_count"
BUILD_SCHEMA_INFERENCE_ERRORS_COUNT = "build.schema_inference_errors_count"

CLI_INVOCATION_ID = "cli.invocation_id"
CLI_COMMAND = "cli.command"
CLI_ARG_COUNT = "cli.arg_count"
CLI_ARG_NAMES = "cli.arg_names"
CLI_DURATION_MS = "cli.duration_ms"
CLI_PARSE_DURATION_MS = "cli.parse_duration_ms"
CLI_EXIT_CODE = "cli.exit_code"
CLI_IS_PARSE_ERROR = "cli.is_parse_error"
CLI_ERROR_TYPE = "cli.error_type"

SHUTDOWN_STATUS = "shutdown.status"
SHUTDOWN_PENDING_TASKS_COUNT = "shutdown.pending_tasks_count"
SHUTDOWN_ACTIVE_THREADS_COUNT = "shutdown.active_threads_count"
SHUTDOWN_SUBPROCESS_COUNT = "shutdown.subprocess_count"
SHUTDOWN_PENDING_TASK_SAMPLES = "shutdown.pending_task_samples"
SHUTDOWN_ACTIVE_THREAD_NAMES = "shutdown.active_thread_names"
SHUTDOWN_SUBPROCESS_SAMPLES = "shutdown.subprocess_samples"
SHUTDOWN_ERROR_TYPE = "shutdown.error_type"
SHUTDOWN_ERROR_MESSAGE = "shutdown.error_message"

SCIP_RUN_ID = "scip.run_id"
SCIP_REPO = "scip.repo"
SCIP_COMMIT = "scip.commit"
SCIP_MODE = "scip.mode"
SCIP_STATUS = "scip.status"
SCIP_ERROR = "scip.error"
SCIP_DURATION_MS = "scip.duration_ms"

TELEMETRY_FLUSH_OK = "telemetry.flush.ok"
TELEMETRY_FLUSH_MS = "telemetry.flush.ms"
TELEMETRY_ACTION = "telemetry.action"
TELEMETRY_INSTRUMENTATION_NAME = "telemetry.instrumentation.name"
TELEMETRY_INSTRUMENTATION_STATUS = "telemetry.instrumentation.status"

__all__ = [
    "BUILD_COMMIT",
    "BUILD_DECISION_TRACE_ARTIFACT",
    "BUILD_DURATION_MS",
    "BUILD_REPO",
    "BUILD_RUN_ID",
    "BUILD_SCHEMA_INFERENCE_ERRORS_COUNT",
    "BUILD_TARGETS",
    "BUILD_VALIDATION_ISSUE_COUNT",
    "BUILD_VALIDATION_MODE",
    "CLI_ARG_COUNT",
    "CLI_ARG_NAMES",
    "CLI_COMMAND",
    "CLI_DURATION_MS",
    "CLI_ERROR_TYPE",
    "CLI_EXIT_CODE",
    "CLI_INVOCATION_ID",
    "CLI_IS_PARSE_ERROR",
    "CLI_PARSE_DURATION_MS",
    "CODEINTEL_ACTOR",
    "CODEINTEL_COMMIT",
    "CODEINTEL_COMPONENT",
    "CODEINTEL_CORRELATION_ID",
    "CODEINTEL_DB_STATEMENT_SHA256",
    "CODEINTEL_DOMAIN",
    "CODEINTEL_ENDPOINT",
    "CODEINTEL_HEALTH_CHECK",
    "CODEINTEL_OPERATION",
    "CODEINTEL_OUTPUT_FORMAT",
    "CODEINTEL_QUERY_ENDPOINT",
    "CODEINTEL_QUERY_HASH",
    "CODEINTEL_QUERY_ROW_COUNT",
    "CODEINTEL_QUERY_SCHEMA_HASH",
    "CODEINTEL_QUERY_TRUNCATED",
    "CODEINTEL_QUERY_VIEW_ID",
    "CODEINTEL_REPO",
    "CODEINTEL_RUN_ID",
    "CODEINTEL_STORAGE_READ_ONLY",
    "CODEINTEL_SUCCESS",
    "DB_NAMESPACE",
    "DB_QUERY_PARAMETER_PREFIX",
    "DB_QUERY_SUMMARY",
    "DB_QUERY_TEXT",
    "DB_STATEMENT",
    "DB_SYSTEM_NAME",
    "HTTP_METHOD",
    "HTTP_ROUTE",
    "MCP_METHOD",
    "MCP_TOOL_NAME",
    "SCIP_COMMIT",
    "SCIP_DURATION_MS",
    "SCIP_ERROR",
    "SCIP_MODE",
    "SCIP_REPO",
    "SCIP_RUN_ID",
    "SCIP_STATUS",
    "SHUTDOWN_ACTIVE_THREADS_COUNT",
    "SHUTDOWN_ACTIVE_THREAD_NAMES",
    "SHUTDOWN_ERROR_MESSAGE",
    "SHUTDOWN_ERROR_TYPE",
    "SHUTDOWN_PENDING_TASKS_COUNT",
    "SHUTDOWN_PENDING_TASK_SAMPLES",
    "SHUTDOWN_STATUS",
    "SHUTDOWN_SUBPROCESS_COUNT",
    "SHUTDOWN_SUBPROCESS_SAMPLES",
    "TELEMETRY_ACTION",
    "TELEMETRY_FLUSH_MS",
    "TELEMETRY_FLUSH_OK",
    "TELEMETRY_INSTRUMENTATION_NAME",
    "TELEMETRY_INSTRUMENTATION_STATUS",
]
