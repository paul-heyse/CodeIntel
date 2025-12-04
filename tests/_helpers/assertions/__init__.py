"""Test assertion helpers.

This module provides reusable assertion functions for test validation.
"""

from __future__ import annotations

from tests._helpers.assertions.common import (
    HasRowCounts,
    HasSuccessAndError,
    assert_failure,
    assert_has_error,
    assert_invalid,
    assert_meta_contains,
    assert_no_error,
    assert_row_count,
    assert_success,
    assert_valid,
    assert_validation_error,
    format_assertion_message,
)
from tests._helpers.assertions.coverage_assertions import assert_single_edge
from tests._helpers.assertions.dataclass_assertions import assert_cannot_setattr
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_true,
)
from tests._helpers.assertions.table_assertions import (
    assert_columns_not_null,
    assert_table_has_rows,
)

__all__ = [
    "HasRowCounts",
    "HasSuccessAndError",
    "assert_cannot_setattr",
    "assert_columns_not_null",
    "assert_failure",
    "assert_has_error",
    "assert_invalid",
    "assert_meta_contains",
    "assert_no_error",
    "assert_row_count",
    "assert_single_edge",
    "assert_success",
    "assert_table_has_rows",
    "assert_valid",
    "assert_validation_error",
    "expect_equal",
    "expect_in",
    "expect_is_instance",
    "expect_length",
    "expect_true",
    "format_assertion_message",
]
