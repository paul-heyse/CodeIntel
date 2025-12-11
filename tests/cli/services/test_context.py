"""Tests for CommandContext and CommandContextBuilder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.context import CommandContextBuilder
from codeintel.cli.rendering.types import OutputFormat


class TestCommandContextBuilder:
    """Test CommandContextBuilder."""

    def test_build_creates_context(self) -> None:
        """Build creates a valid context."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            with CommandContextBuilder().build() as ctx:
                assert ctx is not None
                assert ctx.params is not None
                assert ctx.jobs is not None

    def test_with_params_sets_params(self) -> None:
        """With params sets parameters."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            params = {"name": "test", "count": 42}
            with CommandContextBuilder().with_params(params).build() as ctx:
                assert ctx.params.get_str("name") == "test"
                assert ctx.params.get_int("count") == 42

    def test_with_output_format(self) -> None:
        """With output format sets format."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            builder = CommandContextBuilder().with_output_format(OutputFormat.JSON)
            with builder.build() as ctx:
                assert ctx.output_format == OutputFormat.JSON

    def test_with_verbosity(self) -> None:
        """With verbosity sets level."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            builder = CommandContextBuilder().with_verbosity(2)
            with builder.build() as ctx:
                assert ctx.verbosity == 2

    def test_with_operation_id(self) -> None:
        """With operation ID sets ID."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            builder = CommandContextBuilder().with_operation_id("test.op")
            with builder.build() as ctx:
                assert ctx.operation_id == "test.op"


class TestCommandContext:
    """Test CommandContext."""

    def test_has_runtime_without_config(self) -> None:
        """Has runtime returns False without configuration."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            with CommandContextBuilder().build() as ctx:
                assert not ctx.has_runtime

    def test_has_storage_without_config(self) -> None:
        """Has storage returns False without configuration."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            with CommandContextBuilder().build() as ctx:
                assert not ctx.has_storage

    def test_has_serving_without_config(self) -> None:
        """Has serving returns False without configuration."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            with CommandContextBuilder().build() as ctx:
                assert not ctx.has_serving

    def test_runtime_raises_without_config(self) -> None:
        """Runtime raises error without configuration."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            with CommandContextBuilder().build() as ctx:
                with pytest.raises(RuntimeError, match="Runtime not available"):
                    _ = ctx.runtime

    def test_storage_raises_without_config(self) -> None:
        """Storage raises error without configuration."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            with CommandContextBuilder().build() as ctx:
                with pytest.raises(RuntimeError, match="Storage not available"):
                    _ = ctx.storage

    def test_serving_raises_without_config(self) -> None:
        """Serving raises error without configuration."""
        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            with CommandContextBuilder().build() as ctx:
                with pytest.raises(RuntimeError, match="Serving not available"):
                    _ = ctx.serving


class TestCommandContextWithStorage:
    """Test CommandContext with storage enabled."""

    def test_with_storage_enables_runtime(self, tmp_path: Path) -> None:
        """With storage implicitly enables runtime."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        db_dir = project_dir / "build" / "db"
        db_dir.mkdir(parents=True)

        config_file = project_dir / "codeintel.yaml"
        config_file.write_text("repo: test/repo\nstorage:\n  db_path: build/db/codeintel.duckdb\n")

        with (
            patch("codeintel.cli.context.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            builder = (
                CommandContextBuilder().with_storage().with_params({"project_root": project_dir})
            )
            with builder.build() as ctx:
                assert ctx.has_runtime
                assert ctx.has_storage
