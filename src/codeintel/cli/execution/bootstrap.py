"""CLI bootstrap - single entry point for CLI initialization.

This module provides bootstrap_cli(), the idempotent initialization function
that all CLI entry points should call. It consolidates:

- Logging configuration (from handlers/base.py)
- Signal handler registration
- Configuration loading

Call bootstrap_cli() once at CLI startup. Subsequent calls are no-ops.

WARNING: This module is part of the CLI migration (Phase 1).
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.cli.config import load_config as load_cli_config

if TYPE_CHECKING:
    from types import FrameType

    from codeintel.cli.config.model import CliConfig

LOG = logging.getLogger(__name__)

# Verbosity thresholds (same as handlers/base.py)
VERBOSITY_DEBUG = 2
VERBOSITY_INFO = 1


@dataclass
class _BootstrapState:
    """Internal state for bootstrap management.

    This class encapsulates the bootstrap state to avoid using global
    variables with the global statement.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    complete: bool = False
    config: CliConfig | None = None


# Module-level state instance (singleton)
_state = _BootstrapState()


def bootstrap_cli(
    verbosity: int = 0,
    config: CliConfig | None = None,
) -> CliConfig:
    """Initialize CLI subsystems exactly once.

    This function is idempotent and thread-safe. It should be called at the
    start of every CLI command. Subsequent calls return the cached config.

    Initializes:

    - Logging configuration based on verbosity
    - Signal handlers for graceful shutdown (SIGINT, SIGTERM)

    Parameters
    ----------
    verbosity
        Logging verbosity level:
        - 0 = WARNING (or config default)
        - 1 = INFO
        - 2+ = DEBUG
    config
        Optional pre-loaded configuration. If None, loads from environment.

    Returns
    -------
    CliConfig
        The active CLI configuration.

    Examples
    --------
    >>> from unittest.mock import MagicMock
    >>> reset_bootstrap()  # Ensure clean state for doctest
    >>> mock_config = MagicMock()
    >>> mock_config.log_level = "WARNING"
    >>> result = bootstrap_cli(verbosity=1, config=mock_config)
    >>> result is mock_config
    True
    >>> reset_bootstrap()  # Clean up after doctest
    """
    # Fast path for already initialized
    if _state.complete:
        if _state.config is not None:
            return _state.config
        # Shouldn't happen, but handle gracefully
        return load_cli_config(validate=False)

    with _state.lock:
        # Double-check after acquiring lock
        if _state.complete and _state.config is not None:
            return _state.config

        # Load configuration if not provided
        active_config = config if config is not None else load_cli_config(validate=False)

        # Configure logging
        _configure_logging(verbosity, active_config)

        # Register signal handlers
        _register_signal_handlers()

        # Mark as complete
        _state.config = active_config
        _state.complete = True

        LOG.debug("CLI bootstrap complete (verbosity=%d)", verbosity)

        return active_config


def _configure_logging(verbosity: int, config: CliConfig) -> None:
    """Configure logging based on verbosity.

    Parameters
    ----------
    verbosity
        Verbosity level from CLI.
    config
        CLI configuration.
    """
    level = _determine_log_level(verbosity, config)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,  # Reconfigure if already configured
    )


def _determine_log_level(verbosity: int, config: CliConfig) -> int:
    """Determine log level from verbosity and config.

    Parameters
    ----------
    verbosity
        Verbosity level from CLI.
    config
        CLI configuration.

    Returns
    -------
    int
        Logging level constant.
    """
    if verbosity >= VERBOSITY_DEBUG:
        return logging.DEBUG
    if verbosity >= VERBOSITY_INFO:
        return logging.INFO
    # Use config default
    return getattr(logging, config.log_level, logging.WARNING)


def _register_signal_handlers() -> None:
    """Register signal handlers for graceful shutdown."""
    # Only register on main thread
    if threading.current_thread() is not threading.main_thread():
        return

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        """Handle termination signal."""
        LOG.info("Received signal %d, initiating shutdown", signum)
        sys.exit(128 + signum)

    # Register handlers (ignore if not supported)
    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except (ValueError, OSError):
        # Signal registration may fail in some environments
        LOG.debug("Could not register signal handlers")


def reset_bootstrap() -> None:
    """Reset bootstrap state (for testing only).

    WARNING: This function is for testing purposes only. Do not call
    in production code.

    Examples
    --------
    >>> from unittest.mock import MagicMock
    >>> mock_config = MagicMock()
    >>> mock_config.log_level = "WARNING"
    >>> _ = bootstrap_cli(config=mock_config)
    >>> reset_bootstrap()
    >>> # Now bootstrap can be called again with new config
    """
    with _state.lock:
        _state.complete = False
        _state.config = None


def is_bootstrapped() -> bool:
    """Check if CLI has been bootstrapped.

    Returns
    -------
    bool
        True if bootstrap_cli() has been called successfully.

    Examples
    --------
    >>> reset_bootstrap()
    >>> is_bootstrapped()
    False
    >>> from unittest.mock import MagicMock
    >>> mock_config = MagicMock()
    >>> mock_config.log_level = "WARNING"
    >>> _ = bootstrap_cli(config=mock_config)
    >>> is_bootstrapped()
    True
    >>> reset_bootstrap()
    """
    return _state.complete


__all__ = [
    "VERBOSITY_DEBUG",
    "VERBOSITY_INFO",
    "bootstrap_cli",
    "is_bootstrapped",
    "reset_bootstrap",
]
