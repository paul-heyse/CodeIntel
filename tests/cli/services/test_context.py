"""Tests for CommandContext and CommandContextBuilder."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.context import CommandContextBuilder
from codeintel.cli.rendering.types import OutputFormat
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# CommandContextBuilder
# ---------------------------------------------------------------------------


def test_build_creates_context() -> None:
    """Build creates a valid context."""
    with patch("codeintel.cli.context.load_config") as mock_config:
        mock_config.return_value = MagicMock()

        with CommandContextBuilder().build() as ctx:
            expect_is_not_none(ctx)
            expect_is_not_none(ctx.params)
            expect_is_not_none(ctx.jobs)


def test_with_params_sets_params() -> None:
    """With params sets parameters."""
    with patch("codeintel.cli.context.load_config") as mock_config:
        mock_config.return_value = MagicMock()

        params: dict[str, object] = {"name": "test", "count": 42}
        with CommandContextBuilder().with_params(params).build() as ctx:
            expect_equal(ctx.params.get_str("name"), "test")
            expect_equal(ctx.params.get_int("count"), 42)


def test_with_output_format() -> None:
    """With output format sets format."""
    with patch("codeintel.cli.context.load_config") as mock_config:
        mock_config.return_value = MagicMock()

        builder = CommandContextBuilder().with_output_format(OutputFormat.JSON)
        with builder.build() as ctx:
            expect_equal(ctx.output_format, OutputFormat.JSON)


def test_with_verbosity() -> None:
    """With verbosity sets level."""
    with patch("codeintel.cli.context.load_config") as mock_config:
        mock_config.return_value = MagicMock()

        builder = CommandContextBuilder().with_verbosity(2)
        with builder.build() as ctx:
            expect_equal(ctx.verbosity, 2)


def test_with_operation_id() -> None:
    """With operation ID sets ID."""
    with patch("codeintel.cli.context.load_config") as mock_config:
        mock_config.return_value = MagicMock()

        builder = CommandContextBuilder().with_operation_id("test.op")
        with builder.build() as ctx:
            expect_equal(ctx.operation_id, "test.op")


# ---------------------------------------------------------------------------
# CommandContext
# ---------------------------------------------------------------------------


def test_has_runtime_without_config() -> None:
    """Has runtime returns False without configuration."""
    with (
        patch("codeintel.cli.context.load_config", return_value=MagicMock()),
        CommandContextBuilder().build() as ctx,
    ):
        expect_false(ctx.has_runtime)


def test_has_storage_without_config() -> None:
    """Has storage returns False without configuration."""
    with (
        patch("codeintel.cli.context.load_config", return_value=MagicMock()),
        CommandContextBuilder().build() as ctx,
    ):
        expect_false(ctx.has_storage)


def test_has_serving_without_config() -> None:
    """Has serving returns False without configuration."""
    with (
        patch("codeintel.cli.context.load_config", return_value=MagicMock()),
        CommandContextBuilder().build() as ctx,
    ):
        expect_false(ctx.has_serving)


def test_runtime_raises_without_config() -> None:
    """Runtime raises error without configuration."""
    with (
        patch("codeintel.cli.context.load_config", return_value=MagicMock()),
        CommandContextBuilder().build() as ctx,
        pytest.raises(RuntimeError, match="Runtime not available"),
    ):
        _ = ctx.runtime


def test_storage_raises_without_config() -> None:
    """Storage raises error without configuration."""
    with (
        patch("codeintel.cli.context.load_config", return_value=MagicMock()),
        CommandContextBuilder().build() as ctx,
        pytest.raises(RuntimeError, match="Storage not available"),
    ):
        _ = ctx.storage


def test_serving_raises_without_config() -> None:
    """Serving raises error without configuration."""
    with (
        patch("codeintel.cli.context.load_config", return_value=MagicMock()),
        CommandContextBuilder().build() as ctx,
        pytest.raises(RuntimeError, match="Serving not available"),
    ):
        _ = ctx.serving


# ---------------------------------------------------------------------------
# CommandContext with storage
# ---------------------------------------------------------------------------


def test_with_storage_enables_runtime(tmp_path: Path) -> None:
    """With storage implicitly enables runtime."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    db_dir = project_dir / "build" / "db"
    db_dir.mkdir(parents=True)

    config_file = project_dir / "codeintel.yaml"
    config_file.write_text("repo: test/repo\nstorage:\n  db_path: build/db/codeintel.duckdb\n")

    with patch("codeintel.cli.context.load_config") as mock_config:
        mock_config.return_value = MagicMock()

        builder = CommandContextBuilder().with_storage().with_params({"project_root": project_dir})
        with builder.build() as ctx:
            expect_true(ctx.has_runtime)
            expect_true(ctx.has_storage)
