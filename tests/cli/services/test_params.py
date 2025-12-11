"""Tests for ParamService."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.services.params import ParamError, ParamService


class TestParamServiceGetStr:
    """Test string parameter access."""

    def test_get_str_returns_value(self) -> None:
        """Get existing string parameter."""
        params = ParamService({"name": "test"})
        assert params.get_str("name") == "test"

    def test_get_str_returns_default_when_missing(self) -> None:
        """Return default for missing parameter."""
        params = ParamService({})
        assert params.get_str("name", "default") == "default"

    def test_get_str_converts_non_string(self) -> None:
        """Convert non-string values."""
        params = ParamService({"count": 42})
        assert params.get_str("count") == "42"

    def test_get_str_returns_none_when_missing_no_default(self) -> None:
        """Return None when missing without default."""
        params = ParamService({})
        assert params.get_str("name") is None


class TestParamServiceGetInt:
    """Test integer parameter access."""

    def test_get_int_returns_int_value(self) -> None:
        """Get existing integer parameter."""
        params = ParamService({"count": 42})
        assert params.get_int("count") == 42

    def test_get_int_parses_string(self) -> None:
        """Parse integer from string."""
        params = ParamService({"count": "42"})
        assert params.get_int("count") == 42

    def test_get_int_returns_default_for_invalid(self) -> None:
        """Return default for invalid integer."""
        params = ParamService({"count": "not-a-number"})
        assert params.get_int("count", 10) == 10

    def test_get_int_returns_default_for_bool(self) -> None:
        """Return default for boolean values."""
        params = ParamService({"flag": True})
        assert params.get_int("flag", 5) == 5


class TestParamServiceGetBool:
    """Test boolean parameter access."""

    def test_get_bool_returns_bool_value(self) -> None:
        """Get existing boolean parameter."""
        params = ParamService({"enabled": True})
        assert params.get_bool("enabled") is True

    def test_get_bool_parses_truthy_strings(self) -> None:
        """Parse truthy string values."""
        for value in ["true", "1", "yes", "on", "y"]:
            params = ParamService({"enabled": value})
            assert params.get_bool("enabled") is True

    def test_get_bool_parses_falsy_strings(self) -> None:
        """Parse falsy string values."""
        for value in ["false", "0", "no", "off", "n"]:
            params = ParamService({"enabled": value})
            assert params.get_bool("enabled") is False

    def test_get_bool_returns_default_when_missing(self) -> None:
        """Return default for missing parameter."""
        params = ParamService({})
        assert params.get_bool("enabled", default=True) is True


class TestParamServiceGetPath:
    """Test path parameter access."""

    def test_get_path_returns_path(self) -> None:
        """Get existing path parameter."""
        params = ParamService({"file": "/tmp/test.txt"})
        result = params.get_path("file")
        assert result == Path("/tmp/test.txt")

    def test_get_path_preserves_path_object(self) -> None:
        """Preserve existing Path objects."""
        original = Path("/tmp/test.txt")
        params = ParamService({"file": original})
        assert params.get_path("file") == original

    def test_get_path_returns_default(self) -> None:
        """Return default for missing parameter."""
        default = Path("/default")
        params = ParamService({})
        assert params.get_path("file", default) == default


class TestParamServiceGetEnum:
    """Test enum parameter access."""

    def test_get_enum_returns_enum_value(self) -> None:
        """Get existing enum parameter."""
        params = ParamService({"format": OutputFormat.JSON})
        result = params.get_enum("format", OutputFormat)
        assert result == OutputFormat.JSON

    def test_get_enum_parses_string_by_value(self) -> None:
        """Parse enum from string value."""
        params = ParamService({"format": "json"})
        result = params.get_enum("format", OutputFormat)
        assert result == OutputFormat.JSON

    def test_get_enum_parses_string_by_name(self) -> None:
        """Parse enum from string name."""
        params = ParamService({"format": "JSON"})
        result = params.get_enum("format", OutputFormat)
        assert result == OutputFormat.JSON

    def test_get_enum_returns_default_for_unknown(self) -> None:
        """Return default for unknown value."""
        params = ParamService({"format": "unknown"})
        result = params.get_enum("format", OutputFormat, OutputFormat.TEXT)
        assert result == OutputFormat.TEXT


class TestParamServiceGetList:
    """Test list parameter access."""

    def test_get_list_returns_list(self) -> None:
        """Get existing list parameter."""
        params = ParamService({"items": ["a", "b", "c"]})
        assert params.get_list("items") == ["a", "b", "c"]

    def test_get_list_converts_tuple(self) -> None:
        """Convert tuple to list."""
        params = ParamService({"items": ("a", "b")})
        assert params.get_list("items") == ["a", "b"]

    def test_get_list_wraps_single_value(self) -> None:
        """Wrap single value in list."""
        params = ParamService({"items": "single"})
        assert params.get_list("items") == ["single"]

    def test_get_list_returns_empty_default(self) -> None:
        """Return empty list for missing parameter."""
        params = ParamService({})
        assert params.get_list("items") == []


class TestParamServiceRequire:
    """Test required parameter access."""

    def test_require_str_returns_value(self) -> None:
        """Get required string parameter."""
        params = ParamService({"name": "test"})
        assert params.require_str("name") == "test"

    def test_require_str_raises_for_missing(self) -> None:
        """Raise ParamError for missing required parameter."""
        params = ParamService({})
        with pytest.raises(ParamError) as exc_info:
            params.require_str("name")
        assert exc_info.value.key == "name"
        assert "not provided" in str(exc_info.value)

    def test_require_int_returns_value(self) -> None:
        """Get required integer parameter."""
        params = ParamService({"count": 42})
        assert params.require_int("count") == 42

    def test_require_int_raises_for_invalid(self) -> None:
        """Raise ParamError for invalid integer."""
        params = ParamService({"count": "not-int"})
        with pytest.raises(ParamError) as exc_info:
            params.require_int("count")
        assert "integer" in str(exc_info.value)

    def test_require_path_returns_value(self) -> None:
        """Get required path parameter."""
        params = ParamService({"file": "/tmp/test.txt"})
        assert params.require_path("file") == Path("/tmp/test.txt")

    def test_require_path_raises_for_missing(self) -> None:
        """Raise ParamError for missing path."""
        params = ParamService({})
        with pytest.raises(ParamError):
            params.require_path("file")


class TestParamServiceOutputFormat:
    """Test output format resolution."""

    def test_get_output_format_with_json_flag(self) -> None:
        """JSON flag takes precedence."""
        params = ParamService({"json": True, "output_format": OutputFormat.TEXT})
        assert params.get_output_format() == OutputFormat.JSON

    def test_get_output_format_explicit(self) -> None:
        """Use explicit format when no JSON flag."""
        params = ParamService({"output_format": OutputFormat.JSONL})
        assert params.get_output_format() == OutputFormat.JSONL

    def test_get_output_format_default(self) -> None:
        """Return default when nothing specified."""
        params = ParamService({})
        assert params.get_output_format() == OutputFormat.TEXT


class TestParamServiceCoercion:
    """Test CLI value coercion."""

    def test_coerce_cli_value_int(self) -> None:
        """Coerce integer string."""
        assert ParamService.coerce_cli_value("42") == 42

    def test_coerce_cli_value_float(self) -> None:
        """Coerce float string."""
        assert ParamService.coerce_cli_value("3.14") == 3.14

    def test_coerce_cli_value_bool_true(self) -> None:
        """Coerce truthy string."""
        assert ParamService.coerce_cli_value("true") is True

    def test_coerce_cli_value_bool_false(self) -> None:
        """Coerce falsy string."""
        assert ParamService.coerce_cli_value("false") is False

    def test_coerce_cli_value_string(self) -> None:
        """Return string for non-numeric."""
        assert ParamService.coerce_cli_value("hello") == "hello"


class TestParamServiceMerge:
    """Test parameter merging."""

    def test_merge_creates_new_service(self) -> None:
        """Merge returns new service."""
        original = ParamService({"a": 1})
        merged = original.merge({"b": 2})
        assert merged is not original
        assert merged.get_int("a") == 1
        assert merged.get_int("b") == 2

    def test_merge_overrides_existing(self) -> None:
        """Merged values override originals."""
        original = ParamService({"a": 1})
        merged = original.merge({"a": 2})
        assert merged.get_int("a") == 2
        assert original.get_int("a") == 1  # Original unchanged
