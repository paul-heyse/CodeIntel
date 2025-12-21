"""Canonical error code catalog for CodeIntel serving.

This catalog is transport-agnostic and must remain stable once published.
"""

from __future__ import annotations

from codeintel.core.errors.taxonomy import (
    ALREADY_EXISTS,
    AUTH_FAILED,
    COLUMN_NOT_FOUND,
    CORRUPTION_DETECTED,
    INTERNAL_ERROR,
    INVALID_FORMAT,
    JOB_EXPIRED,
    NOT_FOUND,
    OUT_OF_RANGE,
    QUERY_FAILED,
    SERVICE_UNAVAILABLE,
    TIMEOUT,
)
from codeintel.serving.errors.models import ErrorKind
from codeintel.serving.errors.templates import ErrorInfoTemplate

ERROR_CODE_CATALOG: dict[str, ErrorInfoTemplate] = {
    # -------- Semantic/query layer --------
    "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
        error_code=NOT_FOUND,
        kind=ErrorKind.not_found,
        retryable=False,
        message="Semantic view '{view_id}' not found.",
        hint="Call semantic_catalog to list available views, then retry.",
    ),
    "CODEINTEL_SEMANTIC_INVALID_QUERY": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_INVALID_QUERY",
        error_code=INVALID_FORMAT,
        kind=ErrorKind.invalid_request,
        retryable=False,
        message="Invalid semantic query request.",
        hint="Validate request fields/types and compare against semantic_describe(view_id).",
    ),
    "CODEINTEL_SEMANTIC_INVALID_FILTER": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_INVALID_FILTER",
        error_code=INVALID_FORMAT,
        kind=ErrorKind.invalid_request,
        retryable=False,
        message="Invalid filter specification.",
        hint="Check filter operators and values against the view schema (semantic_describe).",
    ),
    "CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND",
        error_code=COLUMN_NOT_FOUND,
        kind=ErrorKind.invalid_request,
        retryable=False,
        message="Column '{column}' not found in semantic view '{view_id}'.",
        hint="Use semantic_describe(view_id) to list available columns.",
    ),
    "CODEINTEL_SEMANTIC_LIMIT_EXCEEDED": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_LIMIT_EXCEEDED",
        error_code=OUT_OF_RANGE,
        kind=ErrorKind.invalid_request,
        retryable=False,
        message="Requested limit {limit} exceeds maximum {max_limit}.",
        hint="Lower the limit or use export/preview flows for large results.",
    ),
    "CODEINTEL_SEMANTIC_QUERY_TIMEOUT": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_QUERY_TIMEOUT",
        error_code=TIMEOUT,
        kind=ErrorKind.timeout,
        retryable=True,
        message="Semantic query timed out.",
        hint="Retry with narrower filters and/or a smaller limit; use semantic_explain.",
    ),
    "CODEINTEL_SEMANTIC_QUERY_UNAVAILABLE": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_QUERY_UNAVAILABLE",
        error_code=SERVICE_UNAVAILABLE,
        kind=ErrorKind.unavailable,
        retryable=True,
        message="Semantic query backend is temporarily unavailable.",
        hint="Retry shortly. If this persists, check serving snapshot state (serving_meta).",
    ),
    "CODEINTEL_SEMANTIC_INTERNAL_ERROR": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_INTERNAL_ERROR",
        error_code=INTERNAL_ERROR,
        kind=ErrorKind.internal,
        retryable=True,
        message="Internal error while processing semantic request.",
        hint="Retry. If it persists, inspect server logs using debug_id.",
    ),
    # -------- Export subsystem --------
    "CODEINTEL_EXPORT_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_NOT_FOUND",
        error_code=NOT_FOUND,
        kind=ErrorKind.not_found,
        retryable=False,
        message="Export handle '{export_id}' not found.",
        hint="Re-run semantic_query/export to generate a fresh export handle.",
    ),
    "CODEINTEL_EXPORT_EXPIRED": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_EXPIRED",
        error_code=JOB_EXPIRED,
        kind=ErrorKind.expired,
        retryable=True,
        message="Export handle '{export_id}' expired.",
        hint="Re-run semantic_query/export with the same parameters for a fresh handle.",
    ),
    "CODEINTEL_EXPORT_CORRUPT": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_CORRUPT",
        error_code=CORRUPTION_DETECTED,
        kind=ErrorKind.corrupt,
        retryable=True,
        message="Export '{export_id}' is corrupt or unreadable.",
        hint="Re-run semantic_export to regenerate; if it repeats, rebuild snapshot.",
    ),
    "CODEINTEL_EXPORT_TOO_LARGE": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_TOO_LARGE",
        error_code=OUT_OF_RANGE,
        kind=ErrorKind.invalid_request,
        retryable=False,
        message="Export request is too large (rows/bytes exceed server limits).",
        hint="Add filters and/or reduce limit; consider exporting narrower slices.",
    ),
    "CODEINTEL_EXPORT_INVALID_REQUEST": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_INVALID_REQUEST",
        error_code=INVALID_FORMAT,
        kind=ErrorKind.invalid_request,
        retryable=False,
        message="Invalid export request.",
        hint=(
            "Validate view_id, export_format, limit, and filters; "
            "call semantic_describe(view_id) for schema."
        ),
    ),
    "CODEINTEL_EXPORT_UNAVAILABLE": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_UNAVAILABLE",
        error_code=SERVICE_UNAVAILABLE,
        kind=ErrorKind.unavailable,
        retryable=True,
        message="Export store is temporarily unavailable.",
        hint="Retry shortly. If it persists, check disk space and export TTL settings.",
    ),
    "CODEINTEL_EXPORT_INTERNAL_ERROR": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_INTERNAL_ERROR",
        error_code=INTERNAL_ERROR,
        kind=ErrorKind.internal,
        retryable=True,
        message="Internal error while retrieving export.",
        hint="Retry. If it persists, inspect server logs using debug_id.",
    ),
    "CODEINTEL_META_ARTIFACT_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_META_ARTIFACT_NOT_FOUND",
        error_code=NOT_FOUND,
        kind=ErrorKind.not_found,
        retryable=False,
        message="Meta artifact '{artifact}' not found for mounted snapshot.",
        hint="Verify the snapshot was built with serving artifacts enabled.",
    ),
    "CODEINTEL_META_SQL_UNSAFE": ErrorInfoTemplate(
        code="CODEINTEL_META_SQL_UNSAFE",
        error_code=CORRUPTION_DETECTED,
        kind=ErrorKind.corrupt,
        retryable=False,
        message="Meta SQL artifact contains unsafe SQL for view '{view_id}'.",
        hint="Rebuild the snapshot and ensure compiled SQL stays within the select-only perimeter.",
    ),
    # -------- Serving snapshot / DB --------
    "CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED",
        error_code=SERVICE_UNAVAILABLE,
        kind=ErrorKind.unavailable,
        retryable=True,
        message="Serving snapshot is not mounted.",
        hint="Wait for snapshot publication or restart the server; check serving_meta.",
    ),
    "CODEINTEL_SERVING_SNAPSHOT_MISMATCH": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_SNAPSHOT_MISMATCH",
        error_code=ALREADY_EXISTS,
        kind=ErrorKind.conflict,
        retryable=True,
        message="Requested snapshot does not match the mounted serving snapshot.",
        hint="Refresh snapshot metadata (serving_meta) and retry with current identifiers.",
    ),
    "CODEINTEL_SERVING_DB_LOCKED": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_DB_LOCKED",
        error_code=SERVICE_UNAVAILABLE,
        kind=ErrorKind.unavailable,
        retryable=True,
        message="Serving database is busy/locked.",
        hint="Retry shortly (backoff). Consider lowering concurrency.",
    ),
    "CODEINTEL_SERVING_DB_INTERNAL_ERROR": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_DB_INTERNAL_ERROR",
        error_code=QUERY_FAILED,
        kind=ErrorKind.internal,
        retryable=True,
        message="Internal serving database error.",
        hint="Retry. If it persists, inspect server logs using debug_id.",
    ),
    # -------- Schema/spec/auth --------
    "CODEINTEL_SCHEMA_MANIFEST_MISSING": ErrorInfoTemplate(
        code="CODEINTEL_SCHEMA_MANIFEST_MISSING",
        error_code=SERVICE_UNAVAILABLE,
        kind=ErrorKind.unavailable,
        retryable=True,
        message="Schema manifest missing for the mounted snapshot.",
        hint="Rebuild/publish the snapshot or rerun schema compilation.",
    ),
    "CODEINTEL_AUTH_FORBIDDEN": ErrorInfoTemplate(
        code="CODEINTEL_AUTH_FORBIDDEN",
        error_code=AUTH_FAILED,
        kind=ErrorKind.invalid_request,
        retryable=False,
        message="Unauthorized.",
        hint="Provide a valid API key or bearer token, or disable auth enforcement for local use.",
    ),
}

__all__ = ["ERROR_CODE_CATALOG"]
