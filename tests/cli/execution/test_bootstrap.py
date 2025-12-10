"""Tests for CLI bootstrap.

This module provides unit tests for the bootstrap_cli() function
introduced in Phase 1 of the CLI migration.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.execution.bootstrap import (
    bootstrap_cli,
    is_bootstrapped,
    reset_bootstrap,
)


@pytest.fixture(autouse=True)
def _reset_bootstrap_state() -> None:
    """Reset bootstrap state before and after each test."""
    reset_bootstrap()
    yield
    reset_bootstrap()


@pytest.fixture
def mock_config() -> MagicMock:
    """Create mock CLI config."""
    config = MagicMock()
    config.log_level = "WARNING"
    return config


class TestBootstrapCli:
    """Tests for bootstrap_cli function."""

    def test_returns_config(self, mock_config: MagicMock) -> None:
        """Return the provided config."""
        result = bootstrap_cli(config=mock_config)
        assert result is mock_config

    def test_loads_config_when_not_provided(self) -> None:
        """Load config from environment when not provided."""
        with patch(
            "codeintel.cli.execution.bootstrap.load_cli_config",
        ) as mock_load:
            mock_load.return_value = MagicMock(log_level="INFO")
            result = bootstrap_cli()
            mock_load.assert_called_once_with(validate=False)
            assert result is mock_load.return_value

    def test_idempotent_second_call_returns_cached(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Second call returns cached config without re-initialization."""
        first = bootstrap_cli(config=mock_config)

        # Create different config for second call
        other_config = MagicMock(log_level="DEBUG")
        second = bootstrap_cli(config=other_config)

        # Should return first config, not second
        assert second is first
        assert second is mock_config

    def test_configures_logging_at_debug(self, mock_config: MagicMock) -> None:
        """Configure DEBUG logging when verbosity >= 2."""
        bootstrap_cli(verbosity=2, config=mock_config)
        assert logging.getLogger().level == logging.DEBUG

    def test_configures_logging_at_info(self, mock_config: MagicMock) -> None:
        """Configure INFO logging when verbosity == 1."""
        bootstrap_cli(verbosity=1, config=mock_config)
        assert logging.getLogger().level == logging.INFO

    def test_configures_logging_at_warning(self, mock_config: MagicMock) -> None:
        """Configure WARNING logging when verbosity == 0."""
        mock_config.log_level = "WARNING"
        bootstrap_cli(verbosity=0, config=mock_config)
        assert logging.getLogger().level == logging.WARNING


class TestResetBootstrap:
    """Tests for reset_bootstrap function."""

    def test_allows_reinitialize(self, mock_config: MagicMock) -> None:
        """Reset allows re-initialization."""
        first = bootstrap_cli(config=mock_config)
        reset_bootstrap()

        other_config = MagicMock(log_level="DEBUG")
        second = bootstrap_cli(config=other_config)

        # After reset, should use new config
        assert second is other_config
        assert second is not first


class TestIsBootstrapped:
    """Tests for is_bootstrapped function."""

    def test_false_before_bootstrap(self) -> None:
        """Return False before bootstrap_cli is called."""
        assert is_bootstrapped() is False

    def test_true_after_bootstrap(self, mock_config: MagicMock) -> None:
        """Return True after bootstrap_cli is called."""
        bootstrap_cli(config=mock_config)
        assert is_bootstrapped() is True

    def test_false_after_reset(self, mock_config: MagicMock) -> None:
        """Return False after reset_bootstrap is called."""
        bootstrap_cli(config=mock_config)
        assert is_bootstrapped() is True

        reset_bootstrap()
        assert is_bootstrapped() is False
