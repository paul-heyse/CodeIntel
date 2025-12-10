"""Error handling for operations.

Provides RFC 9457 ProblemDetail and factory functions for common errors.
"""

from __future__ import annotations

from codeintel.operations.errors.factory import (
    ErrorResult,
    fail_capability_denied,
    fail_dataset_not_found,
    fail_internal,
    fail_invalid_value,
    fail_job_cancel_failed,
    fail_job_not_completed,
    fail_job_not_found,
    fail_missing_required,
    fail_not_found,
    fail_operation_not_found,
    fail_plugin_not_found,
    fail_storage_connection,
    fail_storage_query,
    fail_validation,
    fail_with_problem,
)
from codeintel.operations.errors.problem_detail import ERROR_TYPE_BASE, ErrorType, ProblemDetail

__all__ = [
    "ERROR_TYPE_BASE",
    "ErrorResult",
    "ErrorType",
    "ProblemDetail",
    "fail_capability_denied",
    "fail_dataset_not_found",
    "fail_internal",
    "fail_invalid_value",
    "fail_job_cancel_failed",
    "fail_job_not_completed",
    "fail_job_not_found",
    "fail_missing_required",
    "fail_not_found",
    "fail_operation_not_found",
    "fail_plugin_not_found",
    "fail_storage_connection",
    "fail_storage_query",
    "fail_validation",
    "fail_with_problem",
]
