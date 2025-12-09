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
    expect_empty,
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_not_empty,
    expect_not_equal,
    expect_not_in,
    expect_true,
    require_row,
    require_rows,
    unwrap_optional,
)
from tests._helpers.assertions.logging_assertions import assert_logged
from tests._helpers.assertions.schema_assertions import (
    assert_mapping_list,
    assert_mapping_value,
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
    "assert_logged",
    "assert_mapping_list",
    "assert_mapping_value",
    "assert_meta_contains",
    "assert_no_error",
    "assert_row_count",
    "assert_single_edge",
    "assert_success",
    "assert_table_has_rows",
    "assert_valid",
    "assert_validation_error",
    "expect_empty",
    "expect_equal",
    "expect_false",
    "expect_in",
    "expect_is_instance",
    "expect_is_none",
    "expect_is_not_none",
    "expect_length",
    "expect_not_empty",
    "expect_not_equal",
    "expect_not_in",
    "expect_true",
    "format_assertion_message",
    "require_row",
    "require_rows",
    "unwrap_optional",
]
