"""Tests for HandlerContext.

This module provides comprehensive unit tests for the unified HandlerContext
introduced in Phase 1 of the CLI migration.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeintel.cli.handlers.context import (
    HandlerContext,
    ParameterError,
    handler_context_manager,
)
from codeintel.cli.rendering.types import OutputFormat


class SampleEnum(Enum):
    """Sample enum for param_enum tests."""

    VALUE_A = "a"
    VALUE_B = "b"


@pytest.fixture
def mock_config() -> MagicMock:
    """Create mock CLI config."""
    config = MagicMock(spec=["output_format", "log_level", "color", "progress"])
    config.output_format = "text"
    config.log_level = "WARNING"
    config.color = True
    return config


@pytest.fixture
def basic_context(mock_config: MagicMock) -> HandlerContext:
    """Create basic HandlerContext for testing."""
    return HandlerContext(
        config=mock_config,
        operation_id="test.operation",
        output_format=OutputFormat.TEXT,
        verbosity=0,
        _params={"name": "test", "count": 10, "enabled": True},
    )


class TestParamStr:
    """Tests for param_str method."""

    def test_returns_value_when_present(self, basic_context: HandlerContext) -> None:
        """Return string value when parameter exists."""
        assert basic_context.param_str("name") == "test"

    def test_returns_default_when_missing(self, basic_context: HandlerContext) -> None:
        """Return default when parameter missing."""
        assert basic_context.param_str("missing", "default") == "default"

    def test_returns_none_when_missing_no_default(
        self,
        basic_context: HandlerContext,
    ) -> None:
        """Return None when parameter missing and no default."""
        assert basic_context.param_str("missing") is None

    def test_converts_int_to_string(self, basic_context: HandlerContext) -> None:
        """Convert non-string values to string."""
        assert basic_context.param_str("count") == "10"


class TestParamInt:
    """Tests for param_int method."""

    def test_returns_value_when_present(self, basic_context: HandlerContext) -> None:
        """Return integer value when parameter exists."""
        assert basic_context.param_int("count") == 10

    def test_returns_default_when_missing(self, basic_context: HandlerContext) -> None:
        """Return default when parameter missing."""
        assert basic_context.param_int("missing", 42) == 42

    def test_converts_string_to_int(self, mock_config: MagicMock) -> None:
        """Convert string values to integer."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"count": "20"},
        )
        assert ctx.param_int("count") == 20

    def test_returns_default_on_invalid(self, mock_config: MagicMock) -> None:
        """Return default when conversion fails."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"count": "not-a-number"},
        )
        assert ctx.param_int("count", 5) == 5

    def test_does_not_treat_bool_as_int(self, mock_config: MagicMock) -> None:
        """Boolean values should not be treated as integers directly."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"flag": True},
        )
        # Bool is converted via str() -> int()
        assert ctx.param_int("flag", 99) == 99  # "True" is not valid int


class TestParamBool:
    """Tests for param_bool method."""

    def test_returns_value_when_present(self, basic_context: HandlerContext) -> None:
        """Return boolean value when parameter exists."""
        assert basic_context.param_bool("enabled") is True

    def test_returns_default_when_missing(self, basic_context: HandlerContext) -> None:
        """Return default when parameter missing."""
        assert basic_context.param_bool("missing", default=True) is True

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_handles_string_values(
        self,
        mock_config: MagicMock,
        value: str,
        expected: bool,
    ) -> None:
        """Handle various string representations of boolean."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"flag": value},
        )
        assert ctx.param_bool("flag") is expected


class TestParamPath:
    """Tests for param_path method."""

    def test_returns_path_when_present(self, mock_config: MagicMock) -> None:
        """Return Path when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"path": Path("/some/path")},
        )
        assert ctx.param_path("path") == Path("/some/path")

    def test_converts_string_to_path(self, mock_config: MagicMock) -> None:
        """Convert string values to Path."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"path": "/some/path"},
        )
        assert ctx.param_path("path") == Path("/some/path")

    def test_returns_default_when_missing(self, mock_config: MagicMock) -> None:
        """Return default when parameter missing."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        assert ctx.param_path("path", Path("/default")) == Path("/default")

    def test_returns_none_when_missing_no_default(self, mock_config: MagicMock) -> None:
        """Return None when parameter missing and no default."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        assert ctx.param_path("path") is None


class TestParamEnum:
    """Tests for param_enum method."""

    def test_returns_enum_when_present(self, mock_config: MagicMock) -> None:
        """Return enum when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"choice": SampleEnum.VALUE_A},
        )
        assert ctx.param_enum("choice", SampleEnum) == SampleEnum.VALUE_A

    def test_converts_string_to_enum(self, mock_config: MagicMock) -> None:
        """Convert string values to enum."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"choice": "a"},
        )
        assert ctx.param_enum("choice", SampleEnum) == SampleEnum.VALUE_A

    def test_returns_default_on_invalid(self, mock_config: MagicMock) -> None:
        """Return default when conversion fails."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"choice": "invalid"},
        )
        assert ctx.param_enum("choice", SampleEnum, SampleEnum.VALUE_B) == SampleEnum.VALUE_B

    def test_returns_none_when_missing_no_default(self, mock_config: MagicMock) -> None:
        """Return None when parameter missing and no default."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        assert ctx.param_enum("choice", SampleEnum) is None


class TestParamList:
    """Tests for param_list method."""

    def test_returns_list_when_present(self, mock_config: MagicMock) -> None:
        """Return list when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"items": ["a", "b", "c"]},
        )
        assert ctx.param_list("items") == ["a", "b", "c"]

    def test_converts_tuple_to_list(self, mock_config: MagicMock) -> None:
        """Convert tuple to list."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"items": ("a", "b")},
        )
        assert ctx.param_list("items") == ["a", "b"]

    def test_returns_empty_list_when_missing(self, mock_config: MagicMock) -> None:
        """Return empty list when parameter missing."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        assert ctx.param_list("items") == []

    def test_returns_default_when_missing(self, mock_config: MagicMock) -> None:
        """Return default when parameter missing."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        assert ctx.param_list("items", ["default"]) == ["default"]


class TestParamTuple:
    """Tests for param_tuple method."""

    def test_returns_tuple_when_present(self, mock_config: MagicMock) -> None:
        """Return tuple when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"items": ("a", "b")},
        )
        assert ctx.param_tuple("items") == ("a", "b")

    def test_converts_list_to_tuple(self, mock_config: MagicMock) -> None:
        """Convert list to tuple."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"items": ["a", "b"]},
        )
        assert ctx.param_tuple("items") == ("a", "b")

    def test_returns_empty_tuple_when_missing(self, mock_config: MagicMock) -> None:
        """Return empty tuple when parameter missing."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        assert ctx.param_tuple("items") == ()


class TestRequireStr:
    """Tests for require_str method."""

    def test_returns_value_when_present(self, basic_context: HandlerContext) -> None:
        """Return string value when parameter exists."""
        assert basic_context.require_str("name") == "test"

    def test_raises_when_missing(self, basic_context: HandlerContext) -> None:
        """Raise ParameterError when parameter missing."""
        with pytest.raises(ParameterError, match="Required parameter 'missing'"):
            basic_context.require_str("missing")


class TestRequireInt:
    """Tests for require_int method."""

    def test_returns_value_when_present(self, basic_context: HandlerContext) -> None:
        """Return integer value when parameter exists."""
        assert basic_context.require_int("count") == 10

    def test_raises_when_missing(self, basic_context: HandlerContext) -> None:
        """Raise ParameterError when parameter missing."""
        with pytest.raises(ParameterError, match="Required parameter 'missing'"):
            basic_context.require_int("missing")

    def test_raises_when_invalid(self, mock_config: MagicMock) -> None:
        """Raise ParameterError when value is not valid integer."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"count": "not-a-number"},
        )
        with pytest.raises(ParameterError, match="must be an integer"):
            ctx.require_int("count")


class TestRequirePath:
    """Tests for require_path method."""

    def test_returns_path_when_present(self, mock_config: MagicMock) -> None:
        """Return Path when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"path": "/some/path"},
        )
        assert ctx.require_path("path") == Path("/some/path")

    def test_raises_when_missing(self, mock_config: MagicMock) -> None:
        """Raise ParameterError when parameter missing."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        with pytest.raises(ParameterError, match="Required parameter 'path'"):
            ctx.require_path("path")


class TestContextManager:
    """Tests for context manager protocol."""

    def test_enter_returns_self(self, basic_context: HandlerContext) -> None:
        """__enter__ returns the context."""
        with basic_context as ctx:
            assert ctx is basic_context

    def test_exit_calls_close(self, mock_config: MagicMock) -> None:
        """__exit__ closes resources."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        with ctx:
            pass
        assert ctx._closed is True

    def test_close_on_exception(self, mock_config: MagicMock) -> None:
        """Resources closed even on exception."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        with pytest.raises(ValueError, match="test error"), ctx:
            msg = "test error"
            raise ValueError(msg)
        assert ctx._closed is True

    def test_close_idempotent(self, mock_config: MagicMock) -> None:
        """Close can be called multiple times safely."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        ctx.close()
        ctx.close()  # Should not raise
        assert ctx._closed is True


class TestHandlerContextManager:
    """Tests for handler_context_manager function."""

    def test_creates_context(self, mock_config: MagicMock) -> None:
        """Create context with correct parameters."""
        from codeintel.cli.handlers.context import HandlerContextOptions

        options = HandlerContextOptions(verbosity=1)
        with handler_context_manager(
            mock_config,
            "test.op",
            params={"key": "value"},
            options=options,
        ) as ctx:
            assert ctx.operation_id == "test.op"
            assert ctx.verbosity == 1
            assert ctx.param_str("key") == "value"

    def test_closes_on_exit(self, mock_config: MagicMock) -> None:
        """Close context on exit."""
        with handler_context_manager(mock_config, "test.op") as ctx:
            pass
        assert ctx._closed is True

    def test_closes_on_exception(self, mock_config: MagicMock) -> None:
        """Close context on exception."""
        with pytest.raises(ValueError, match="test"):
            with handler_context_manager(mock_config, "test.op") as ctx:
                msg = "test"
                raise ValueError(msg)
        assert ctx._closed is True


class TestConvenienceProperties:
    """Tests for convenience properties."""

    def test_logger_property(self, mock_config: MagicMock) -> None:
        """Logger property returns correct logger."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="my.operation",
        )
        logger = ctx.logger
        assert logger.name == "codeintel.cli.handlers.my.operation"

    def test_color_enabled(self, mock_config: MagicMock) -> None:
        """Color enabled returns config value."""
        mock_config.color = True
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        assert ctx.color_enabled is True

        mock_config.color = False
        assert ctx.color_enabled is False

    def test_db_path_returns_none_without_runtime(self, mock_config: MagicMock) -> None:
        """Db path returns None without runtime or database_path."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        assert ctx.db_path is None

    def test_db_path_returns_database_path(self, mock_config: MagicMock) -> None:
        """Db path returns database_path when set."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            database_path=Path("/test/db.duckdb"),
        )
        assert ctx.db_path == Path("/test/db.duckdb")
