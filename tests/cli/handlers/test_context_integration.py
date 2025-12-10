"""Integration tests for new HandlerContext.

These tests verify end-to-end workflows without touching legacy code paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.rendering.types import OutputFormat


class TestHandlerContextIntegration:
    """Integration tests for HandlerContext."""

    def test_full_param_workflow(self) -> None:
        """Test complete parameter workflow."""
        config = MagicMock()
        config.log_level = "WARNING"
        config.color = True

        ctx = HandlerContext(
            config=config,
            operation_id="test.integration",
            output_format=OutputFormat.JSON,
            verbosity=1,
            project_root=Path("/test/project"),
            _params={
                "name": "test-name",
                "count": 42,
                "enabled": True,
                "path": "/test/path",
            },
        )

        # Test all param accessors
        assert ctx.param_str("name") == "test-name"
        assert ctx.param_int("count") == 42
        assert ctx.param_bool("enabled") is True
        assert ctx.param_path("path") == Path("/test/path")

        # Test require variants
        assert ctx.require_str("name") == "test-name"
        assert ctx.require_int("count") == 42

        # Test context properties
        assert ctx.operation_id == "test.integration"
        assert ctx.output_format == OutputFormat.JSON
        assert ctx.verbosity == 1
        assert ctx.project_root == Path("/test/project")

    def test_context_manager_closes_resources(self) -> None:
        """Test context manager properly closes resources."""
        config = MagicMock()
        config.log_level = "WARNING"

        with HandlerContext(
            config=config,
            operation_id="test.cleanup",
            _params={},
        ) as ctx:
            assert ctx._closed is False

        assert ctx._closed is True

    def test_logger_property(self) -> None:
        """Test logger property returns correct logger."""
        config = MagicMock()
        config.log_level = "WARNING"

        ctx = HandlerContext(
            config=config,
            operation_id="my.operation",
            _params={},
        )

        logger = ctx.logger
        assert logger.name == "codeintel.cli.handlers.my.operation"

    def test_nested_contexts(self) -> None:
        """Test nested context managers work correctly."""
        config = MagicMock()
        config.log_level = "WARNING"

        with HandlerContext(
            config=config,
            operation_id="outer",
            _params={"level": "outer"},
        ) as outer:
            assert outer.param_str("level") == "outer"

            with HandlerContext(
                config=config,
                operation_id="inner",
                _params={"level": "inner"},
            ) as inner:
                assert inner.param_str("level") == "inner"
                assert outer.param_str("level") == "outer"

            # Inner should be closed
            assert inner._closed is True
            # Outer still open
            assert outer._closed is False

        # Both closed
        assert outer._closed is True
        assert inner._closed is True

    def test_output_format_variations(self) -> None:
        """Test context with different output formats."""
        config = MagicMock()
        config.log_level = "WARNING"

        for fmt in [OutputFormat.TEXT, OutputFormat.JSON, OutputFormat.JSONL]:
            ctx = HandlerContext(
                config=config,
                operation_id="test.format",
                output_format=fmt,
                _params={},
            )
            assert ctx.output_format == fmt

    def test_verbosity_levels(self) -> None:
        """Test context with different verbosity levels."""
        config = MagicMock()
        config.log_level = "WARNING"

        for level in [0, 1, 2, 3]:
            ctx = HandlerContext(
                config=config,
                operation_id="test.verbosity",
                verbosity=level,
                _params={},
            )
            assert ctx.verbosity == level

    def test_database_path_fallback(self) -> None:
        """Test database_path is accessible via db_path property."""
        config = MagicMock()
        config.log_level = "WARNING"

        db_path = Path("/test/db.duckdb")
        ctx = HandlerContext(
            config=config,
            operation_id="test.db",
            database_path=db_path,
            _params={},
        )

        assert ctx.db_path == db_path
