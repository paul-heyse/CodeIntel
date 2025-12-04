"""Tests for profile utility functions.

This module tests:
- optional_str conversion
- optional_int conversion
- int_or_default with fallback
- optional_float conversion
- optional_bool conversion
"""

from __future__ import annotations

import pytest

from codeintel.analytics.profiles.utils import (
    CATALOG_MODULE_TABLE,
    DEFAULT_MODULE_TABLE,
    int_or_default,
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
)

# Test constants
DEFAULT_INT_VALUE = 0
CUSTOM_DEFAULT_INT = 42
TEST_INT_VALUE = 123
TEST_FLOAT_VALUE = 3.14
TEST_STRING_INT_456 = 456
TEST_STRING_INT_789 = 789
TRUNCATED_FLOAT_TO_INT = 3
FLOAT_TOLERANCE = 0.00001
TEST_FLOAT_2_5 = 2.5


class TestOptionalStr:
    """Tests for optional_str function."""

    @staticmethod
    def test_returns_string_for_string_input() -> None:
        """Verify string input returns same string."""
        assert optional_str("hello") == "hello"

    @staticmethod
    def test_returns_string_for_int_input() -> None:
        """Verify int input is converted to string."""
        assert optional_str(TEST_INT_VALUE) == "123"

    @staticmethod
    def test_returns_string_for_float_input() -> None:
        """Verify float input is converted to string."""
        result = optional_str(TEST_FLOAT_VALUE)
        assert result is not None
        assert "3.14" in result

    @staticmethod
    def test_returns_none_for_none_input() -> None:
        """Verify None input returns None."""
        assert optional_str(None) is None

    @staticmethod
    def test_returns_string_for_bool_true_input() -> None:
        """Verify True input is converted to string."""
        bool_true: bool = True
        assert optional_str(bool_true) == "True"

    @staticmethod
    def test_returns_string_for_bool_false_input() -> None:
        """Verify False input is converted to string."""
        bool_false: bool = False
        assert optional_str(bool_false) == "False"


class TestOptionalInt:
    """Tests for optional_int function."""

    @staticmethod
    def test_returns_int_for_int_input() -> None:
        """Verify int input returns same int."""
        assert optional_int(TEST_INT_VALUE) == TEST_INT_VALUE

    @staticmethod
    def test_returns_int_for_float_input() -> None:
        """Verify float input is truncated to int."""
        assert optional_int(TEST_FLOAT_VALUE) == TRUNCATED_FLOAT_TO_INT

    @staticmethod
    def test_returns_int_for_string_number() -> None:
        """Verify numeric string is converted to int."""
        assert optional_int("456") == TEST_STRING_INT_456

    @staticmethod
    def test_returns_int_for_string_with_whitespace() -> None:
        """Verify string with whitespace is converted."""
        assert optional_int("  789  ") == TEST_STRING_INT_789

    @staticmethod
    def test_returns_none_for_empty_string() -> None:
        """Verify empty string returns None."""
        assert optional_int("") is None

    @staticmethod
    def test_returns_none_for_whitespace_only_string() -> None:
        """Verify whitespace-only string returns None."""
        assert optional_int("   ") is None

    @staticmethod
    def test_returns_none_for_invalid_string() -> None:
        """Verify non-numeric string returns None."""
        assert optional_int("abc") is None

    @staticmethod
    def test_returns_none_for_none_input() -> None:
        """Verify None input returns None."""
        assert optional_int(None) is None

    @staticmethod
    def test_returns_int_for_bool_true() -> None:
        """Verify True is converted to 1."""
        bool_true: bool = True
        assert optional_int(bool_true) == 1

    @staticmethod
    def test_returns_int_for_bool_false() -> None:
        """Verify False is converted to 0."""
        bool_false: bool = False
        assert optional_int(bool_false) == 0


class TestIntOrDefault:
    """Tests for int_or_default function."""

    @staticmethod
    def test_returns_int_for_int_input() -> None:
        """Verify int input returns same int."""
        assert int_or_default(TEST_INT_VALUE) == TEST_INT_VALUE

    @staticmethod
    def test_returns_default_for_none() -> None:
        """Verify None returns default."""
        assert int_or_default(None) == DEFAULT_INT_VALUE

    @staticmethod
    def test_returns_custom_default_for_none() -> None:
        """Verify None returns custom default."""
        assert int_or_default(None, CUSTOM_DEFAULT_INT) == CUSTOM_DEFAULT_INT

    @staticmethod
    def test_returns_default_for_invalid_string() -> None:
        """Verify invalid string returns default."""
        assert int_or_default("abc") == DEFAULT_INT_VALUE

    @staticmethod
    def test_returns_int_for_valid_string() -> None:
        """Verify valid string returns int."""
        assert int_or_default("42") == CUSTOM_DEFAULT_INT


class TestOptionalFloat:
    """Tests for optional_float function."""

    @staticmethod
    def test_returns_float_for_float_input() -> None:
        """Verify float input returns same float."""
        assert optional_float(TEST_FLOAT_VALUE) == TEST_FLOAT_VALUE

    @staticmethod
    def test_returns_float_for_int_input() -> None:
        """Verify int input is converted to float."""
        result = optional_float(TEST_INT_VALUE)
        assert result is not None
        assert result == float(TEST_INT_VALUE)

    @staticmethod
    def test_returns_float_for_string_number() -> None:
        """Verify numeric string is converted to float."""
        expected = 3.14159
        result = optional_float("3.14159")
        assert result is not None
        assert abs(result - expected) < FLOAT_TOLERANCE

    @staticmethod
    def test_returns_float_for_string_with_whitespace() -> None:
        """Verify string with whitespace is converted."""
        result = optional_float("  2.5  ")
        assert result is not None
        assert result == TEST_FLOAT_2_5

    @staticmethod
    def test_returns_none_for_empty_string() -> None:
        """Verify empty string returns None."""
        assert optional_float("") is None

    @staticmethod
    def test_returns_none_for_whitespace_only_string() -> None:
        """Verify whitespace-only string returns None."""
        assert optional_float("   ") is None

    @staticmethod
    def test_returns_none_for_invalid_string() -> None:
        """Verify non-numeric string returns None."""
        assert optional_float("abc") is None

    @staticmethod
    def test_returns_none_for_none_input() -> None:
        """Verify None input returns None."""
        assert optional_float(None) is None

    @staticmethod
    def test_returns_float_for_bool_true() -> None:
        """Verify True is converted to 1.0."""
        bool_true: bool = True
        assert optional_float(bool_true) == 1.0

    @staticmethod
    def test_returns_float_for_bool_false() -> None:
        """Verify False is converted to 0.0."""
        bool_false: bool = False
        assert optional_float(bool_false) == 0.0


class TestOptionalBool:
    """Tests for optional_bool function."""

    @staticmethod
    def test_returns_bool_for_bool_true() -> None:
        """Verify True returns True."""
        bool_true: bool = True
        assert optional_bool(bool_true) is True

    @staticmethod
    def test_returns_bool_for_bool_false() -> None:
        """Verify False returns False."""
        bool_false: bool = False
        assert optional_bool(bool_false) is False

    @staticmethod
    def test_returns_bool_for_int_nonzero() -> None:
        """Verify nonzero int returns True."""
        one = 1
        assert optional_bool(one) is True
        assert optional_bool(TEST_INT_VALUE) is True

    @staticmethod
    def test_returns_bool_for_int_zero() -> None:
        """Verify zero int returns False."""
        zero = 0
        assert optional_bool(zero) is False

    @staticmethod
    def test_returns_bool_for_float_nonzero() -> None:
        """Verify nonzero float returns True."""
        one_float = 1.0
        assert optional_bool(one_float) is True
        assert optional_bool(TEST_FLOAT_VALUE) is True

    @staticmethod
    def test_returns_bool_for_float_zero() -> None:
        """Verify zero float returns False."""
        zero_float = 0.0
        assert optional_bool(zero_float) is False

    @staticmethod
    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "YES"])
    def test_returns_true_for_truthy_strings(value: str) -> None:
        """Verify truthy strings return True."""
        assert optional_bool(value) is True

    @staticmethod
    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "NO"])
    def test_returns_false_for_falsy_strings(value: str) -> None:
        """Verify falsy strings return False."""
        assert optional_bool(value) is False

    @staticmethod
    def test_returns_true_for_string_with_whitespace() -> None:
        """Verify whitespace is stripped before checking."""
        assert optional_bool("  true  ") is True
        assert optional_bool("  false  ") is False

    @staticmethod
    def test_returns_none_for_empty_string() -> None:
        """Verify empty string returns None."""
        assert optional_bool("") is None

    @staticmethod
    def test_returns_none_for_invalid_string() -> None:
        """Verify invalid string returns None."""
        assert optional_bool("maybe") is None
        assert optional_bool("abc") is None

    @staticmethod
    def test_returns_none_for_none_input() -> None:
        """Verify None input returns None."""
        assert optional_bool(None) is None


class TestModuleTableConstants:
    """Tests for module table constants."""

    @staticmethod
    def test_catalog_module_table_value() -> None:
        """Verify CATALOG_MODULE_TABLE constant."""
        assert CATALOG_MODULE_TABLE == "temp.catalog_modules"

    @staticmethod
    def test_default_module_table_value() -> None:
        """Verify DEFAULT_MODULE_TABLE constant."""
        assert DEFAULT_MODULE_TABLE == "core.modules"
