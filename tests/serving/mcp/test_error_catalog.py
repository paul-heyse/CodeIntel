"""Tests for MCP error catalog and infrastructure.

These tests ensure:
1. All expected error codes exist in the catalog (contract test)
2. Error codes match their catalog keys (consistency)
3. Helper functions produce valid responses
4. Domain exceptions convert to correct error codes
"""

from __future__ import annotations

from codeintel.serving.errors import (
    ERROR_CODE_CATALOG,
    AuthForbiddenError,
    ErrorContext,
    ErrorInfo,
    ErrorKind,
    ErrorResponse,
    ExportCorruptError,
    ExportExpiredError,
    ExportNotFoundError,
    ExportTooLargeError,
    SemanticColumnNotFoundError,
    SemanticInvalidFilterError,
    SemanticLimitExceededError,
    SemanticViewNotFoundError,
    ServingDBLockedError,
    ServingSnapshotNotMountedError,
    exception_to_error_response,
)
from codeintel.serving.errors.mapping import error_from_code
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_not_none,
    expect_true,
)

# =============================================================================
# Contract Tests - Lock in the error codes
# =============================================================================

EXPECTED_CODES = {
    "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
    "CODEINTEL_SEMANTIC_INVALID_QUERY",
    "CODEINTEL_SEMANTIC_INVALID_FILTER",
    "CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND",
    "CODEINTEL_SEMANTIC_LIMIT_EXCEEDED",
    "CODEINTEL_SEMANTIC_QUERY_TIMEOUT",
    "CODEINTEL_SEMANTIC_QUERY_UNAVAILABLE",
    "CODEINTEL_SEMANTIC_INTERNAL_ERROR",
    "CODEINTEL_EXPORT_INVALID_REQUEST",
    "CODEINTEL_EXPORT_NOT_FOUND",
    "CODEINTEL_EXPORT_EXPIRED",
    "CODEINTEL_EXPORT_CORRUPT",
    "CODEINTEL_EXPORT_TOO_LARGE",
    "CODEINTEL_EXPORT_UNAVAILABLE",
    "CODEINTEL_EXPORT_INTERNAL_ERROR",
    "CODEINTEL_META_ARTIFACT_NOT_FOUND",
    "CODEINTEL_META_SQL_UNSAFE",
    "CODEINTEL_SEARCH_INDEX_MISSING",
    "CODEINTEL_LINEAGE_MISSING",
    "CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED",
    "CODEINTEL_SERVING_SNAPSHOT_MISMATCH",
    "CODEINTEL_SERVING_DB_LOCKED",
    "CODEINTEL_SERVING_DB_INTERNAL_ERROR",
    "CODEINTEL_SCHEMA_MANIFEST_MISSING",
    "CODEINTEL_AUTH_FORBIDDEN",
}


def test_error_catalog_codes_are_locked_in() -> None:
    """Verify all expected error codes exist in catalog."""
    expect_equal(set(ERROR_CODE_CATALOG.keys()), EXPECTED_CODES)


def test_error_catalog_has_25_codes() -> None:
    """Verify catalog contains exactly 25 codes."""
    expect_equal(len(ERROR_CODE_CATALOG), 25)


def test_error_catalog_codes_match_keys() -> None:
    """Verify each template's code matches its catalog key."""
    for key, tmpl in ERROR_CODE_CATALOG.items():
        expect_equal(tmpl.code, key)


def test_error_catalog_all_have_messages() -> None:
    """Verify all error codes have non-empty messages."""
    for code, tmpl in ERROR_CODE_CATALOG.items():
        expect_true(len(tmpl.message) > 0, message=f"{code} should have a message")


def test_error_catalog_all_have_error_codes() -> None:
    """Verify all error codes map to the canonical taxonomy."""
    for code, tmpl in ERROR_CODE_CATALOG.items():
        expect_is_not_none(tmpl.error_code, message=f"{code} should have error_code")


# =============================================================================
# Error Model Tests
# =============================================================================


def test_error_kind_has_expected_values() -> None:
    """Verify ErrorKind enum has all expected values."""
    expected = {
        "invalid_request",
        "not_found",
        "expired",
        "corrupt",
        "conflict",
        "unavailable",
        "timeout",
        "internal",
    }
    actual = {k.value for k in ErrorKind}
    expect_equal(actual, expected)


def test_error_info_model_validates() -> None:
    """Verify ErrorInfo model accepts valid data."""
    info = ErrorInfo(
        code="CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
        kind=ErrorKind.not_found,
        message="View not found",
        retryable=False,
        hint="Check semantic_catalog",
        details={"view_id": "test"},
    )
    expect_equal(info.code, "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND")
    expect_equal(info.kind, ErrorKind.not_found)
    expect_false(info.retryable)


def test_error_response_model_validates() -> None:
    """Verify ErrorResponse model wraps ErrorInfo correctly."""
    response = ErrorResponse(
        error=ErrorInfo(
            code="CODEINTEL_EXPORT_NOT_FOUND",
            kind=ErrorKind.not_found,
            message="Export not found",
            hint=None,
        )
    )
    expect_equal(response.status, "error")
    expect_equal(response.error.code, "CODEINTEL_EXPORT_NOT_FOUND")


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_error_from_code_returns_valid_response() -> None:
    """Verify error_from_code produces valid ErrorResponse."""
    response = error_from_code("CODEINTEL_SEMANTIC_VIEW_NOT_FOUND")
    expect_equal(response.status, "error")
    expect_equal(response.error.code, "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND")
    expect_equal(response.error.kind, ErrorKind.not_found)
    expect_false(response.error.retryable)


def test_error_from_code_with_params() -> None:
    """Verify error_from_code substitutes template parameters."""
    response = error_from_code(
        "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
        params={"view_id": "test_view"},
    )
    expect_in("test_view", response.error.message)


def test_error_from_code_with_context() -> None:
    """Verify error_from_code includes context in details."""
    ctx = ErrorContext(
        operation="semantic_query",
        tool_name="semantic_query",
        view_id="test_view",
    )
    response = error_from_code(
        "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
        context=ctx,
        params={"view_id": "test_view"},
    )
    expect_equal(response.error.details["operation"], "semantic_query")
    expect_equal(response.error.details["view_id"], "test_view")
    expect_in("debug_id", response.error.details)
    expect_in("ts", response.error.details)


def test_error_from_code_with_extra_details() -> None:
    """Verify error_from_code merges extra details."""
    response = error_from_code(
        "CODEINTEL_SEMANTIC_INTERNAL_ERROR",
        details={"custom_field": "custom_value"},
    )
    expect_equal(response.error.details["custom_field"], "custom_value")


def test_error_from_code_unknown_code_falls_back() -> None:
    """Verify unknown codes fall back to internal error."""
    response = error_from_code("UNKNOWN_CODE")
    expect_equal(response.error.code, "CODEINTEL_SEMANTIC_INTERNAL_ERROR")


def test_error_from_code_retryable_preserved() -> None:
    """Verify retryable flag is preserved from catalog."""
    response_retryable = error_from_code("CODEINTEL_SEMANTIC_QUERY_TIMEOUT")
    expect_true(response_retryable.error.retryable)

    response_not_retryable = error_from_code("CODEINTEL_SEMANTIC_VIEW_NOT_FOUND")
    expect_false(response_not_retryable.error.retryable)


# =============================================================================
# Exception Mapper Tests
# =============================================================================


def test_exception_to_error_response_timeout() -> None:
    """Verify TimeoutError maps to query timeout."""
    ctx = ErrorContext(operation="semantic_query")
    response = exception_to_error_response(TimeoutError(), context=ctx)
    expect_equal(response.error.code, "CODEINTEL_SEMANTIC_QUERY_TIMEOUT")
    expect_true(response.error.retryable)


def test_exception_to_error_response_keyerror_with_view_id() -> None:
    """Verify KeyError with view_id context maps to view not found."""
    ctx = ErrorContext(operation="semantic_query", view_id="missing_view")
    response = exception_to_error_response(KeyError("missing_view"), context=ctx)
    expect_equal(response.error.code, "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND")


def test_exception_to_error_response_generic_exception() -> None:
    """Verify generic exceptions map to internal error."""
    ctx = ErrorContext(operation="semantic_query")
    response = exception_to_error_response(RuntimeError("oops"), context=ctx)
    expect_equal(response.error.code, "CODEINTEL_SEMANTIC_INTERNAL_ERROR")
    expect_equal(response.error.details["exception_type"], "RuntimeError")


def test_exception_to_error_response_domain_exception() -> None:
    """Verify domain exceptions use their own code."""
    ctx = ErrorContext(operation="semantic_query")
    exc = SemanticViewNotFoundError("my_view")
    response = exception_to_error_response(exc, context=ctx)
    expect_equal(response.error.code, "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND")
    expect_in("my_view", response.error.message)


# =============================================================================
# Domain Exception Tests
# =============================================================================


def test_semantic_view_not_found_error() -> None:
    """Verify SemanticViewNotFoundError exception."""
    exc = SemanticViewNotFoundError("test_view")
    expect_equal(exc.code, "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND")
    expect_equal(exc.params["view_id"], "test_view")


def test_semantic_column_not_found_error() -> None:
    """Verify SemanticColumnNotFoundError exception."""
    exc = SemanticColumnNotFoundError("test_view", "missing_col")
    expect_equal(exc.code, "CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND")
    expect_equal(exc.params["view_id"], "test_view")
    expect_equal(exc.params["column"], "missing_col")


def test_semantic_invalid_filter_error() -> None:
    """Verify SemanticInvalidFilterError exception."""
    exc = SemanticInvalidFilterError(reason="bad operator")
    expect_equal(exc.code, "CODEINTEL_SEMANTIC_INVALID_FILTER")
    expect_equal(exc.details["reason"], "bad operator")


def test_semantic_limit_exceeded_error() -> None:
    """Verify SemanticLimitExceededError exception."""
    exc = SemanticLimitExceededError(limit=10000, max_limit=5000)
    expect_equal(exc.code, "CODEINTEL_SEMANTIC_LIMIT_EXCEEDED")
    expect_equal(exc.params["limit"], 10000)
    expect_equal(exc.params["max_limit"], 5000)


def test_export_not_found_error() -> None:
    """Verify ExportNotFoundError exception."""
    exc = ExportNotFoundError("export123")
    expect_equal(exc.code, "CODEINTEL_EXPORT_NOT_FOUND")
    expect_equal(exc.params["export_id"], "export123")


def test_export_expired_error() -> None:
    """Verify ExportExpiredError exception."""
    exc = ExportExpiredError("export123", expires_at="2024-01-01T00:00:00Z")
    expect_equal(exc.code, "CODEINTEL_EXPORT_EXPIRED")
    expect_equal(exc.params["export_id"], "export123")
    expect_equal(exc.details["expires_at"], "2024-01-01T00:00:00Z")


def test_export_corrupt_error() -> None:
    """Verify ExportCorruptError exception."""
    exc = ExportCorruptError("export123")
    expect_equal(exc.code, "CODEINTEL_EXPORT_CORRUPT")
    expect_equal(exc.params["export_id"], "export123")


def test_export_too_large_error() -> None:
    """Verify ExportTooLargeError exception."""
    exc = ExportTooLargeError(row_count=1000000)
    expect_equal(exc.code, "CODEINTEL_EXPORT_TOO_LARGE")
    expect_equal(exc.details["row_count"], 1000000)


def test_serving_snapshot_not_mounted_error() -> None:
    """Verify ServingSnapshotNotMountedError exception."""
    exc = ServingSnapshotNotMountedError()
    expect_equal(exc.code, "CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED")


def test_serving_db_locked_error() -> None:
    """Verify ServingDBLockedError exception."""
    exc = ServingDBLockedError()
    expect_equal(exc.code, "CODEINTEL_SERVING_DB_LOCKED")


def test_auth_forbidden_error() -> None:
    """Verify AuthForbiddenError exception."""
    exc = AuthForbiddenError(reason="remote access disabled")
    expect_equal(exc.code, "CODEINTEL_AUTH_FORBIDDEN")
    expect_equal(exc.details["reason"], "remote access disabled")


def test_domain_exception_to_error_response() -> None:
    """Verify domain exceptions convert via to_error_response."""
    exc = SemanticLimitExceededError(limit=10000, max_limit=5000)
    ctx = ErrorContext(operation="semantic_query", tool_name="semantic_query")
    response = exc.to_error_response(context=ctx)

    expect_equal(response.status, "error")
    expect_equal(response.error.code, "CODEINTEL_SEMANTIC_LIMIT_EXCEEDED")
    expect_in("10000", response.error.message)
    expect_in("5000", response.error.message)
    expect_equal(response.error.details["operation"], "semantic_query")


# =============================================================================
# Template Rendering Tests
# =============================================================================


def test_template_renders_with_missing_params() -> None:
    """Verify templates gracefully handle missing parameters."""
    response = error_from_code(
        "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
        params={},  # Missing view_id
    )
    # Should keep {view_id} placeholder instead of crashing
    expect_in("{view_id}", response.error.message)


def test_template_renders_hint() -> None:
    """Verify hint templates are rendered."""
    response = error_from_code(
        "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
        params={"view_id": "test"},
    )
    expect_is_not_none(response.error.hint)
    expect_in("semantic_catalog", response.error.hint or "")


def test_error_response_json_serialization() -> None:
    """Verify ErrorResponse serializes to JSON correctly."""
    response = error_from_code(
        "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
        params={"view_id": "test"},
    )
    json_data = response.model_dump(mode="json")
    expect_equal(json_data["status"], "error")
    expect_equal(json_data["error"]["code"], "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND")
    expect_equal(json_data["error"]["kind"], "not_found")
