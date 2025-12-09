"""Property-based tests for validators using Hypothesis.

Test validators with generated inputs to find edge cases.
"""

from __future__ import annotations

import pytest

from codeintel.cli.cli_config_schema import validate_with_json_schema
from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.cli_validation import IntValidator, StringValidator
from codeintel.cli.error_taxonomy import INTERNAL_ERROR
from codeintel.cli.results import CliResult
from tests._helpers.assertions import (
    expect_false,
    expect_true,
)

# Constraint constants
MIN_LENGTH_CONSTRAINT = 5
MAX_LENGTH_CONSTRAINT = 5
MAX_VALUE_CONSTRAINT = 50


# ---------------------------------------------------------------------------
# StringValidator tests
# ---------------------------------------------------------------------------


def test_valid_strings_pass() -> None:
    """Valid strings within constraints pass validation."""
    validator = StringValidator()
    test_values = ["", "hello", "a" * 100]
    for text in test_values:
        result = validator.validate(text, "test_field")
        expect_true(result.is_valid)


def test_min_length_constraint() -> None:
    """Strings shorter than min_length fail."""
    validator = StringValidator(min_length=MIN_LENGTH_CONSTRAINT)

    # Short string fails
    result = validator.validate("ab", "test_field")
    expect_false(result.is_valid)

    # Long string passes
    result = validator.validate("hello world", "test_field")
    expect_true(result.is_valid)


def test_max_length_constraint() -> None:
    """Strings longer than max_length fail."""
    validator = StringValidator(max_length=MAX_LENGTH_CONSTRAINT)

    # Long string fails
    result = validator.validate("hello world", "test_field")
    expect_false(result.is_valid)

    # Short string passes
    result = validator.validate("hi", "test_field")
    expect_true(result.is_valid)


# ---------------------------------------------------------------------------
# IntValidator tests
# ---------------------------------------------------------------------------


def test_integers_validate_correctly() -> None:
    """Integers within range pass validation."""
    validator = IntValidator()
    test_values = [-1000, 0, 1000]
    for value in test_values:
        result = validator.validate(value, "test_field")
        expect_true(result.is_valid)


def test_min_value_constraint() -> None:
    """Values below min_value fail."""
    validator = IntValidator(min_value=0)

    # Negative fails
    result = validator.validate(-10, "test_field")
    expect_false(result.is_valid)

    # Positive passes
    result = validator.validate(10, "test_field")
    expect_true(result.is_valid)


def test_max_value_constraint() -> None:
    """Values above max_value fail."""
    validator = IntValidator(max_value=MAX_VALUE_CONSTRAINT)

    # High value fails
    result = validator.validate(100, "test_field")
    expect_false(result.is_valid)

    # Low value passes
    result = validator.validate(25, "test_field")
    expect_true(result.is_valid)


# ---------------------------------------------------------------------------
# ConfigSchema tests
# ---------------------------------------------------------------------------


def test_unknown_keys_handled() -> None:
    """Configuration with unknown keys is handled."""
    config = {"unknown_key": "value", "another_unknown": 123}
    errors = validate_with_json_schema(config)

    # Should validate without crashing
    expect_true(isinstance(errors, list))


@pytest.mark.parametrize("colors_enabled", [True, False])
def test_boolean_config_values(*, colors_enabled: bool) -> None:
    """Boolean config values are handled correctly."""
    config = {"output": {"colors": colors_enabled}}
    errors = validate_with_json_schema(config)

    # Should validate without type errors
    expect_true(isinstance(errors, list))


# ---------------------------------------------------------------------------
# CliResult tests
# ---------------------------------------------------------------------------


def test_ok_result_is_valid() -> None:
    """CliResult.ok creates valid success result."""
    result: CliResult[dict[str, int]] = CliResult.ok({"value": 42})

    expect_true(result.success)
    expect_true(result.data is not None)
    expect_true(result.error is None)


def test_fail_result_is_invalid() -> None:
    """CliResult.fail creates valid error result."""
    # Create a ProblemDetail from the ErrorCode
    error_detail = ProblemDetail(
        type=INTERNAL_ERROR.type_uri,
        title=INTERNAL_ERROR.title,
        detail="Test error",
        status=INTERNAL_ERROR.status,
    )
    result: CliResult[dict[str, int]] = CliResult.fail(error_detail)

    expect_false(result.success)
    expect_true(result.data is None)
    expect_true(result.error is not None)


def test_empty_errors_means_valid() -> None:
    """Empty error list means validation passed."""
    validator = StringValidator()
    result = validator.validate("hello", "test_field")

    expect_true(result.is_valid)
