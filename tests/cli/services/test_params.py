"""Tests for ParamService."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.services.params import ParamError, ParamService
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_none,
    expect_is_not,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_get_str_returns_value() -> None:
    """Get existing string parameter."""
    params = ParamService({"name": "test"})
    expect_equal(params.get_str("name"), "test")


def test_get_str_returns_default_when_missing() -> None:
    """Return default for missing parameter."""
    params = ParamService({})
    expect_equal(params.get_str("name", "default"), "default")


def test_get_str_converts_non_string() -> None:
    """Convert non-string values."""
    params = ParamService({"count": 42})
    expect_equal(params.get_str("count"), "42")


def test_get_str_returns_none_when_missing_no_default() -> None:
    """Return None when missing without default."""
    params = ParamService({})
    expect_is_none(params.get_str("name"))


def test_get_int_returns_int_value() -> None:
    """Get existing integer parameter."""
    params = ParamService({"count": 42})
    expect_equal(params.get_int("count"), 42)


def test_get_int_parses_string() -> None:
    """Parse integer from string."""
    params = ParamService({"count": "42"})
    expect_equal(params.get_int("count"), 42)


def test_get_int_returns_default_for_invalid() -> None:
    """Return default for invalid integer."""
    params = ParamService({"count": "not-a-number"})
    expect_equal(params.get_int("count", 10), 10)


def test_get_int_returns_default_for_bool() -> None:
    """Return default for boolean values."""
    params = ParamService({"flag": True})
    expect_equal(params.get_int("flag", 5), 5)


def test_get_bool_returns_bool_value() -> None:
    """Get existing boolean parameter."""
    params = ParamService({"enabled": True})
    expect_true(params.get_bool("enabled"))


def test_get_bool_parses_truthy_strings() -> None:
    """Parse truthy string values."""
    for value in ["true", "1", "yes", "on", "y"]:
        params = ParamService({"enabled": value})
        expect_true(params.get_bool("enabled"))


def test_get_bool_parses_falsy_strings() -> None:
    """Parse falsy string values."""
    for value in ["false", "0", "no", "off", "n"]:
        params = ParamService({"enabled": value})
        expect_false(params.get_bool("enabled"))


def test_get_bool_returns_default_when_missing() -> None:
    """Return default for missing parameter."""
    params = ParamService({})
    expect_true(params.get_bool("enabled", default=True))


def test_get_path_returns_path(tmp_path: Path) -> None:
    """Get existing path parameter."""
    file_path = tmp_path / "test.txt"
    params = ParamService({"file": file_path})
    result = params.get_path("file")
    expect_equal(result, file_path)


def test_get_path_preserves_path_object(tmp_path: Path) -> None:
    """Preserve existing Path objects."""
    original = tmp_path / "test.txt"
    params = ParamService({"file": original})
    expect_equal(params.get_path("file"), original)


def test_get_path_returns_default(tmp_path: Path) -> None:
    """Return default for missing parameter."""
    default = tmp_path / "default"
    params = ParamService({})
    expect_equal(params.get_path("file", default), default)


def test_get_enum_returns_enum_value() -> None:
    """Get existing enum parameter."""
    params = ParamService({"format": OutputFormat.JSON})
    result = params.get_enum("format", OutputFormat)
    expect_equal(result, OutputFormat.JSON)


def test_get_enum_parses_string_by_value() -> None:
    """Parse enum from string value."""
    params = ParamService({"format": "json"})
    result = params.get_enum("format", OutputFormat)
    expect_equal(result, OutputFormat.JSON)


def test_get_enum_parses_string_by_name() -> None:
    """Parse enum from string name."""
    params = ParamService({"format": "JSON"})
    result = params.get_enum("format", OutputFormat)
    expect_equal(result, OutputFormat.JSON)


def test_get_enum_returns_default_for_unknown() -> None:
    """Return default for unknown value."""
    params = ParamService({"format": "unknown"})
    result = params.get_enum("format", OutputFormat, OutputFormat.TEXT)
    expect_equal(result, OutputFormat.TEXT)


def test_get_list_returns_list() -> None:
    """Get existing list parameter."""
    params = ParamService({"items": ["a", "b", "c"]})
    expect_equal(params.get_list("items"), ["a", "b", "c"])


def test_get_list_converts_tuple() -> None:
    """Convert tuple to list."""
    params = ParamService({"items": ("a", "b")})
    expect_equal(params.get_list("items"), ["a", "b"])


def test_get_list_wraps_single_value() -> None:
    """Wrap single value in list."""
    params = ParamService({"items": "single"})
    expect_equal(params.get_list("items"), ["single"])


def test_get_list_returns_empty_default() -> None:
    """Return empty list for missing parameter."""
    params = ParamService({})
    expect_equal(params.get_list("items"), [])


def test_require_str_returns_value() -> None:
    """Get required string parameter."""
    params = ParamService({"name": "test"})
    expect_equal(params.require_str("name"), "test")


def test_require_str_raises_for_missing() -> None:
    """Raise ParamError for missing required parameter."""
    params = ParamService({})
    with pytest.raises(ParamError) as exc_info:
        params.require_str("name")
    expect_equal(exc_info.value.key, "name")
    expect_in("not provided", str(exc_info.value))


def test_require_int_returns_value() -> None:
    """Get required integer parameter."""
    params = ParamService({"count": 42})
    expect_equal(params.require_int("count"), 42)


def test_require_int_raises_for_invalid() -> None:
    """Raise ParamError for invalid integer."""
    params = ParamService({"count": "not-int"})
    with pytest.raises(ParamError) as exc_info:
        params.require_int("count")
    expect_in("integer", str(exc_info.value))


def test_require_path_returns_value(tmp_path: Path) -> None:
    """Get required path parameter."""
    file_path = tmp_path / "test.txt"
    params = ParamService({"file": file_path})
    expect_equal(params.require_path("file"), file_path)


def test_require_path_raises_for_missing() -> None:
    """Raise ParamError for missing path."""
    params = ParamService({})
    with pytest.raises(ParamError):
        params.require_path("file")


def test_get_output_format_with_json_flag() -> None:
    """JSON flag takes precedence."""
    params = ParamService({"json": True, "output_format": OutputFormat.TEXT})
    expect_equal(params.get_output_format(), OutputFormat.JSON)


def test_get_output_format_explicit() -> None:
    """Use explicit format when no JSON flag."""
    params = ParamService({"output_format": OutputFormat.JSONL})
    expect_equal(params.get_output_format(), OutputFormat.JSONL)


def test_get_output_format_default() -> None:
    """Return default when nothing specified."""
    params = ParamService({})
    expect_equal(params.get_output_format(), OutputFormat.TEXT)


def test_coerce_cli_value_int() -> None:
    """Coerce integer string."""
    expect_equal(ParamService.coerce_cli_value("42"), 42)


def test_coerce_cli_value_float() -> None:
    """Coerce float string."""
    expect_equal(ParamService.coerce_cli_value("3.14"), 3.14)


def test_coerce_cli_value_bool_true() -> None:
    """Coerce truthy string."""
    expect_true(ParamService.coerce_cli_value("true") is True)


def test_coerce_cli_value_bool_false() -> None:
    """Coerce falsy string."""
    expect_false(ParamService.coerce_cli_value("false"))


def test_coerce_cli_value_string() -> None:
    """Return string for non-numeric."""
    expect_equal(ParamService.coerce_cli_value("hello"), "hello")


def test_merge_creates_new_service() -> None:
    """Merge returns new service."""
    original = ParamService({"a": 1})
    merged = original.merge({"b": 2})
    expect_is_not(merged, original)
    expect_equal(merged.get_int("a"), 1)
    expect_equal(merged.get_int("b"), 2)


def test_merge_overrides_existing() -> None:
    """Merged values override originals."""
    original = ParamService({"a": 1})
    merged = original.merge({"a": 2})
    expect_equal(merged.get_int("a"), 2)
    expect_equal(original.get_int("a"), 1)
