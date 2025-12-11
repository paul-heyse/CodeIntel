"""Tests for CLI bootstrap.

This module provides unit tests for the bootstrap_cli() function
introduced in Phase 1 of the CLI migration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.config.model import CliConfig
from codeintel.cli.execution.bootstrap import (
    bootstrap_cli,
    is_bootstrapped,
    reset_bootstrap,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _reset_bootstrap_state() -> Generator[None]:
    """Reset bootstrap state before and after each test.

    Yields
    ------
    None
        Ensures bootstrap state is reset around each test invocation.
    """
    reset_bootstrap()
    yield
    reset_bootstrap()


@pytest.fixture
def mock_config() -> CliConfig:
    """Create mock CLI config.

    Returns
    -------
    CliConfig
        Mocked configuration object with default log level.
    """
    config = MagicMock(spec=CliConfig)
    config.log_level = "WARNING"
    # Mock the telemetry attribute accessed by bootstrap_cli
    config.telemetry = MagicMock()
    config.telemetry.enabled = False
    return cast("CliConfig", config)


class TestBootstrapCli:
    """Tests for bootstrap_cli function."""

    @staticmethod
    def test_returns_config(mock_config: CliConfig) -> None:
        """Return the provided config."""
        result = bootstrap_cli(config=mock_config)
        expect_true(result is mock_config)

    @staticmethod
    def test_loads_config_when_not_provided() -> None:
        """Load config from environment when not provided."""
        with patch(
            "codeintel.cli.execution.bootstrap.load_cli_config",
        ) as mock_load:
            mock_config = MagicMock(log_level="INFO")
            mock_config.telemetry = MagicMock()
            mock_config.telemetry.enabled = False
            mock_load.return_value = mock_config
            result = bootstrap_cli()
            mock_load.assert_called_once_with(validate=False)
            expect_true(result is mock_load.return_value)

    @staticmethod
    def test_idempotent_second_call_returns_cached(
        mock_config: CliConfig,
    ) -> None:
        """Second call returns cached config without re-initialization."""
        first = bootstrap_cli(config=mock_config)

        # Create different config for second call
        other_config = MagicMock(log_level="DEBUG")
        other_config.telemetry = MagicMock()
        other_config.telemetry.enabled = False
        second = bootstrap_cli(config=other_config)

        # Should return first config, not second
        expect_true(second is first)
        expect_true(second is mock_config)

    @staticmethod
    def test_configures_logging_at_debug(mock_config: CliConfig) -> None:
        """Configure DEBUG logging when verbosity >= 2."""
        bootstrap_cli(verbosity=2, config=mock_config)
        expect_equal(logging.getLogger().level, logging.DEBUG)

    @staticmethod
    def test_configures_logging_at_info(mock_config: CliConfig) -> None:
        """Configure INFO logging when verbosity == 1."""
        bootstrap_cli(verbosity=1, config=mock_config)
        expect_equal(logging.getLogger().level, logging.INFO)

    @staticmethod
    def test_configures_logging_at_warning(mock_config: CliConfig) -> None:
        """Configure WARNING logging when verbosity == 0."""
        bootstrap_cli(verbosity=0, config=mock_config)
        expect_equal(logging.getLogger().level, logging.WARNING)


class TestResetBootstrap:
    """Tests for reset_bootstrap function."""

    @staticmethod
    def test_allows_reinitialize(mock_config: CliConfig) -> None:
        """Reset allows re-initialization."""
        first = bootstrap_cli(config=mock_config)
        reset_bootstrap()

        other_config = MagicMock(log_level="DEBUG")
        other_config.telemetry = MagicMock()
        other_config.telemetry.enabled = False
        second = bootstrap_cli(config=other_config)

        # After reset, should use new config
        expect_true(second is other_config)
        expect_false(second is first)


class TestIsBootstrapped:
    """Tests for is_bootstrapped function."""

    @staticmethod
    def test_false_before_bootstrap() -> None:
        """Return False before bootstrap_cli is called."""
        expect_false(is_bootstrapped())

    @staticmethod
    def test_true_after_bootstrap(mock_config: CliConfig) -> None:
        """Return True after bootstrap_cli is called."""
        bootstrap_cli(config=mock_config)
        expect_true(is_bootstrapped())

    @staticmethod
    def test_false_after_reset(mock_config: CliConfig) -> None:
        """Return False after reset_bootstrap is called."""
        bootstrap_cli(config=mock_config)
        expect_true(is_bootstrapped())

        reset_bootstrap()
        expect_false(is_bootstrapped())
