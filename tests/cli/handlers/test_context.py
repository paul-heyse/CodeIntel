"""Tests for HandlerContext.

This module provides comprehensive unit tests for the unified HandlerContext
introduced in Phase 1 of the CLI migration.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from codeintel.cli.config.model import CliConfig
from codeintel.cli.handlers.context import (
    HandlerContext,
    HandlerContextOptions,
    ParameterError,
    handler_context_manager,
)
from codeintel.cli.rendering.types import OutputFormat
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_is_none,
    expect_true,
)


class SampleEnum(Enum):
    """Sample enum for param_enum tests."""

    VALUE_A = "a"
    VALUE_B = "b"


@pytest.fixture
def mock_config() -> CliConfig:
    """Create mock CLI config.

    Returns
    -------
    CliConfig
        Mocked configuration object with default CLI settings.
    """
    config = MagicMock(spec=CliConfig)
    config.output_format = "text"
    config.log_level = "WARNING"
    config.color = True
    config.progress = False
    return cast("CliConfig", config)


@pytest.fixture
def basic_context(mock_config: CliConfig) -> HandlerContext:
    """Create basic HandlerContext for testing.

    Returns
    -------
    HandlerContext
        Context configured with default parameters for unit tests.
    """
    return HandlerContext(
        config=mock_config,
        operation_id="test.operation",
        output_format=OutputFormat.TEXT,
        verbosity=0,
        _params={"name": "test", "count": 10, "enabled": True},
    )


class TestParamStr:
    """Tests for param_str method."""

    @staticmethod
    def test_returns_value_when_present(basic_context: HandlerContext) -> None:
        """Return string value when parameter exists."""
        expect_equal(basic_context.param_str("name"), "test")

    @staticmethod
    def test_returns_default_when_missing(basic_context: HandlerContext) -> None:
        """Return default when parameter missing."""
        expect_equal(basic_context.param_str("missing", "default"), "default")

    @staticmethod
    def test_returns_none_when_missing_no_default(
        basic_context: HandlerContext,
    ) -> None:
        """Return None when parameter missing and no default."""
        expect_is_none(basic_context.param_str("missing"))

    @staticmethod
    def test_converts_int_to_string(basic_context: HandlerContext) -> None:
        """Convert non-string values to string."""
        expect_equal(basic_context.param_str("count"), "10")


class TestParamInt:
    """Tests for param_int method."""

    @staticmethod
    def test_returns_value_when_present(basic_context: HandlerContext) -> None:
        """Return integer value when parameter exists."""
        expect_equal(basic_context.param_int("count"), 10)

    @staticmethod
    def test_returns_default_when_missing(basic_context: HandlerContext) -> None:
        """Return default when parameter missing."""
        expect_equal(basic_context.param_int("missing", 42), 42)

    @staticmethod
    def test_converts_string_to_int(mock_config: CliConfig) -> None:
        """Convert string values to integer."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"count": "20"},
        )
        expect_equal(ctx.param_int("count"), 20)

    @staticmethod
    def test_returns_default_on_invalid(mock_config: CliConfig) -> None:
        """Return default when conversion fails."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"count": "not-a-number"},
        )
        expect_equal(ctx.param_int("count", 5), 5)

    @staticmethod
    def test_does_not_treat_bool_as_int(mock_config: CliConfig) -> None:
        """Boolean values should not be treated as integers directly."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"flag": True},
        )
        # Bool is converted via str() -> int()
        expect_equal(ctx.param_int("flag", 99), 99)  # "True" is not valid int


class TestParamBool:
    """Tests for param_bool method."""

    @staticmethod
    def test_returns_value_when_present(basic_context: HandlerContext) -> None:
        """Return boolean value when parameter exists."""
        expect_true(basic_context.param_bool("enabled"))

    @staticmethod
    def test_returns_default_when_missing(basic_context: HandlerContext) -> None:
        """Return default when parameter missing."""
        expect_true(basic_context.param_bool("missing", default=True))

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
    @staticmethod
    def test_handles_string_values(
        mock_config: CliConfig,
        value: str,
        *,
        expected: bool,
    ) -> None:
        """Handle various string representations of boolean."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"flag": value},
        )
        expect_equal(ctx.param_bool("flag"), expected)


class TestParamPath:
    """Tests for param_path method."""

    @staticmethod
    def test_returns_path_when_present(mock_config: CliConfig) -> None:
        """Return Path when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"path": Path("/some/path")},
        )
        expect_equal(ctx.param_path("path"), Path("/some/path"))

    @staticmethod
    def test_converts_string_to_path(mock_config: CliConfig) -> None:
        """Convert string values to Path."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"path": "/some/path"},
        )
        expect_equal(ctx.param_path("path"), Path("/some/path"))

    @staticmethod
    def test_returns_default_when_missing(mock_config: CliConfig) -> None:
        """Return default when parameter missing."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        expect_equal(ctx.param_path("path", Path("/default")), Path("/default"))

    @staticmethod
    def test_returns_none_when_missing_no_default(mock_config: CliConfig) -> None:
        """Return None when parameter missing and no default."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        expect_is_none(ctx.param_path("path"))


class TestParamEnum:
    """Tests for param_enum method."""

    @staticmethod
    def test_returns_enum_when_present(mock_config: CliConfig) -> None:
        """Return enum when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"choice": SampleEnum.VALUE_A},
        )
        expect_equal(ctx.param_enum("choice", SampleEnum), SampleEnum.VALUE_A)

    @staticmethod
    def test_converts_string_to_enum(mock_config: CliConfig) -> None:
        """Convert string values to enum."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"choice": "a"},
        )
        expect_equal(ctx.param_enum("choice", SampleEnum), SampleEnum.VALUE_A)

    @staticmethod
    def test_returns_default_on_invalid(mock_config: CliConfig) -> None:
        """Return default when conversion fails."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"choice": "invalid"},
        )
        expect_equal(ctx.param_enum("choice", SampleEnum, SampleEnum.VALUE_B), SampleEnum.VALUE_B)

    @staticmethod
    def test_returns_none_when_missing_no_default(mock_config: CliConfig) -> None:
        """Return None when parameter missing and no default."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        expect_is_none(ctx.param_enum("choice", SampleEnum))


class TestParamList:
    """Tests for param_list method."""

    @staticmethod
    def test_returns_list_when_present(mock_config: CliConfig) -> None:
        """Return list when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"items": ["a", "b", "c"]},
        )
        expect_equal(ctx.param_list("items"), ["a", "b", "c"])

    @staticmethod
    def test_converts_tuple_to_list(mock_config: CliConfig) -> None:
        """Convert tuple to list."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"items": ("a", "b")},
        )
        expect_equal(ctx.param_list("items"), ["a", "b"])

    @staticmethod
    def test_returns_empty_list_when_missing(mock_config: CliConfig) -> None:
        """Return empty list when parameter missing."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        expect_equal(ctx.param_list("items"), [])

    @staticmethod
    def test_returns_default_when_missing(mock_config: CliConfig) -> None:
        """Return default when parameter missing."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        expect_equal(ctx.param_list("items", ["default"]), ["default"])


class TestParamTuple:
    """Tests for param_tuple method."""

    @staticmethod
    def test_returns_tuple_when_present(mock_config: CliConfig) -> None:
        """Return tuple when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"items": ("a", "b")},
        )
        expect_equal(ctx.param_tuple("items"), ("a", "b"))

    @staticmethod
    def test_converts_list_to_tuple(mock_config: CliConfig) -> None:
        """Convert list to tuple."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"items": ["a", "b"]},
        )
        expect_equal(ctx.param_tuple("items"), ("a", "b"))

    @staticmethod
    def test_returns_empty_tuple_when_missing(mock_config: CliConfig) -> None:
        """Return empty tuple when parameter missing."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={},
        )
        expect_equal(ctx.param_tuple("items"), ())


class TestRequireStr:
    """Tests for require_str method."""

    @staticmethod
    def test_returns_value_when_present(basic_context: HandlerContext) -> None:
        """Return string value when parameter exists."""
        expect_equal(basic_context.require_str("name"), "test")

    @staticmethod
    def test_raises_when_missing(basic_context: HandlerContext) -> None:
        """Raise ParameterError when parameter missing."""
        with pytest.raises(ParameterError, match="Required parameter 'missing'"):
            basic_context.require_str("missing")


class TestRequireInt:
    """Tests for require_int method."""

    @staticmethod
    def test_returns_value_when_present(basic_context: HandlerContext) -> None:
        """Return integer value when parameter exists."""
        expect_equal(basic_context.require_int("count"), 10)

    @staticmethod
    def test_raises_when_missing(basic_context: HandlerContext) -> None:
        """Raise ParameterError when parameter missing."""
        with pytest.raises(ParameterError, match="Required parameter 'missing'"):
            basic_context.require_int("missing")

    @staticmethod
    def test_raises_when_invalid(mock_config: CliConfig) -> None:
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

    @staticmethod
    def test_returns_path_when_present(mock_config: CliConfig) -> None:
        """Return Path when parameter exists."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            _params={"path": "/some/path"},
        )
        expect_equal(ctx.require_path("path"), Path("/some/path"))

    @staticmethod
    def test_raises_when_missing(mock_config: CliConfig) -> None:
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

    @staticmethod
    def test_enter_returns_self(basic_context: HandlerContext) -> None:
        """__enter__ returns the context."""
        with basic_context as ctx:
            expect_true(ctx is basic_context)

    @staticmethod
    def test_exit_calls_close(mock_config: CliConfig) -> None:
        """__exit__ closes resources."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        with ctx:
            pass
        expect_true(ctx.is_closed)

    @staticmethod
    def test_close_on_exception(mock_config: CliConfig) -> None:
        """Resources closed even on exception.

        Raises
        ------
        ValueError
            Raised intentionally to verify cleanup on error.
        """
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        with ctx:
            with pytest.raises(ValueError, match="test error"):
                raise ValueError("test error")
        expect_true(ctx.is_closed)

    @staticmethod
    def test_close_idempotent(mock_config: CliConfig) -> None:
        """Close can be called multiple times safely."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        ctx.close()
        ctx.close()  # Should not raise
        expect_true(ctx.is_closed)


class TestHandlerContextManager:
    """Tests for handler_context_manager function."""

    @staticmethod
    def test_creates_context(mock_config: CliConfig) -> None:
        """Create context with correct parameters."""
        options = HandlerContextOptions(verbosity=1)
        with handler_context_manager(
            mock_config,
            "test.op",
            params={"key": "value"},
            options=options,
        ) as ctx:
            expect_equal(ctx.operation_id, "test.op")
            expect_equal(ctx.verbosity, 1)
            expect_equal(ctx.param_str("key"), "value")

    @staticmethod
    def test_closes_on_exit(mock_config: CliConfig) -> None:
        """Close context on exit."""
        with handler_context_manager(mock_config, "test.op") as ctx:
            pass
        expect_true(ctx.is_closed)

    @staticmethod
    def test_closes_on_exception(mock_config: CliConfig) -> None:
        """Close context on exception.

        Raises
        ------
        ValueError
            Raised intentionally to verify cleanup on error.
        """
        with handler_context_manager(mock_config, "test.op") as ctx, pytest.raises(
            ValueError, match="test"
        ):
            raise ValueError("test")
        expect_true(ctx.is_closed)


class TestConvenienceProperties:
    """Tests for convenience properties."""

    @staticmethod
    def test_logger_property(mock_config: CliConfig) -> None:
        """Logger property returns correct logger."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="my.operation",
        )
        logger = ctx.logger
        expect_equal(logger.name, "codeintel.cli.handlers.my.operation")

    @staticmethod
    def test_color_enabled(mock_config: CliConfig) -> None:
        """Color enabled returns config value."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        expect_true(ctx.color_enabled)

        expect_false(ctx.color_enabled)

    @staticmethod
    def test_db_path_returns_none_without_runtime(mock_config: CliConfig) -> None:
        """Db path returns None without runtime or database_path."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
        )
        expect_is_none(ctx.db_path)

    @staticmethod
    def test_db_path_returns_database_path(mock_config: CliConfig) -> None:
        """Db path returns database_path when set."""
        ctx = HandlerContext(
            config=mock_config,
            operation_id="test",
            database_path=Path("/test/db.duckdb"),
        )
        expect_equal(ctx.db_path, Path("/test/db.duckdb"))
