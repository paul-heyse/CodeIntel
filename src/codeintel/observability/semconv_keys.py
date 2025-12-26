"""Semantic attribute key constants for observability."""

from __future__ import annotations

CODEINTEL_COMPONENT = "codeintel.component"
CODEINTEL_OPERATION = "codeintel.operation"
CODEINTEL_SUCCESS = "codeintel.success"
CODEINTEL_ENDPOINT = "codeintel.endpoint"
CODEINTEL_OUTPUT_FORMAT = "codeintel.output_format"

CODEINTEL_CORRELATION_ID = "codeintel.correlation_id"
CODEINTEL_RUN_ID = "codeintel.run_id"
CODEINTEL_DOMAIN = "codeintel.domain"
CODEINTEL_REPO = "codeintel.repo"
CODEINTEL_COMMIT = "codeintel.commit"

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

TELEMETRY_FLUSH_OK = "telemetry.flush.ok"
TELEMETRY_FLUSH_MS = "telemetry.flush.ms"

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
    "CLI_COMMAND",
    "CLI_ERROR_TYPE",
    "CLI_EXIT_CODE",
    "CLI_INVOCATION_ID",
    "CLI_IS_PARSE_ERROR",
    "CODEINTEL_COMMIT",
    "CODEINTEL_COMPONENT",
    "CODEINTEL_CORRELATION_ID",
    "CODEINTEL_DB_STATEMENT_SHA256",
    "CODEINTEL_DOMAIN",
    "CODEINTEL_ENDPOINT",
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
    "SHUTDOWN_ACTIVE_THREADS_COUNT",
    "SHUTDOWN_ACTIVE_THREAD_NAMES",
    "SHUTDOWN_PENDING_TASKS_COUNT",
    "SHUTDOWN_PENDING_TASK_SAMPLES",
    "SHUTDOWN_STATUS",
    "SHUTDOWN_SUBPROCESS_COUNT",
    "SHUTDOWN_SUBPROCESS_SAMPLES",
    "TELEMETRY_FLUSH_MS",
    "TELEMETRY_FLUSH_OK",
]
