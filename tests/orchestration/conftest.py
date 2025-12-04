"""Prefect test fixtures for quiet, graceful orchestration runs."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

# Prefect server API has a Python 3.13 compatibility issue with string annotation
# evaluation. The `provide_database_interface` function uses TYPE_CHECKING-guarded
# imports for type hints that aren't available at runtime for inspect.signature
# evaluation with eval_str=True. Skip Prefect-dependent tests until the issue is fixed.
# See: https://github.com/PrefectHQ/prefect/issues/XXXXX (to be filed)
_PREFECT_PYTHON_313_ISSUE = sys.version_info >= (3, 13)

# Only import Prefect server components on Python < 3.13
_prefect_server = None
_temporary_settings = None
_PREFECT_API_KEY = None
_PREFECT_API_URL = None
_prefect_test_harness = None

if not _PREFECT_PYTHON_313_ISSUE:
    try:
        from prefect.server.api import server as _prefect_server

        # Prefect settings are runtime-dynamic, pyrefly can't verify them
        from prefect.settings import (
            PREFECT_API_KEY as _PREFECT_API_KEY,  # pyrefly: ignore[missing-module-attribute]
        )
        from prefect.settings import (
            PREFECT_API_URL as _PREFECT_API_URL,  # pyrefly: ignore[missing-module-attribute]
        )
        from prefect.settings import temporary_settings as _temporary_settings
        from prefect.testing.utilities import prefect_test_harness as _prefect_test_harness
    except ImportError:
        # Prefect not installed or import failed
        pass


@contextmanager
def _quiet_prefect_logging() -> Iterator[None]:
    """Suppress Prefect/Rich console logging to avoid noisy teardown errors."""
    if _prefect_server is None:
        yield
        return

    logging.disable(logging.CRITICAL)
    prefect_logger = logging.getLogger("prefect")
    subprocess_logger = _prefect_server.subprocess_server_logger

    pref_handlers = list(prefect_logger.handlers)
    pref_propagate = prefect_logger.propagate
    sub_handlers = list(subprocess_logger.handlers)
    sub_level = subprocess_logger.level
    sub_propagate = subprocess_logger.propagate

    prefect_logger.handlers = [logging.NullHandler()]
    prefect_logger.propagate = False
    subprocess_logger.handlers = [logging.NullHandler()]
    subprocess_logger.setLevel(logging.CRITICAL)
    subprocess_logger.propagate = False

    try:
        yield
    finally:
        prefect_logger.handlers = pref_handlers
        prefect_logger.propagate = pref_propagate
        subprocess_logger.handlers = sub_handlers
        subprocess_logger.setLevel(sub_level)
        subprocess_logger.propagate = sub_propagate
        logging.disable(logging.NOTSET)


@pytest.fixture
def prefect_quiet_env() -> Iterator[None]:
    """Run Prefect flows against the ephemeral test harness with minimal logging.

    The harness starts a temporary API/database and ensures clean shutdown,
    preventing teardown errors or CRASHED states when the process exits.

    Note
    ----
    This fixture is skipped on Python 3.13+ due to Prefect server API
    incompatibility with string annotation evaluation.
    """
    # Check if Prefect is available
    prefect_unavailable = (
        _prefect_server is None
        or _temporary_settings is None
        or _prefect_test_harness is None
    )
    prefect_settings_unavailable = _PREFECT_API_URL is None or _PREFECT_API_KEY is None

    if _PREFECT_PYTHON_313_ISSUE or prefect_unavailable or prefect_settings_unavailable:
        pytest.skip(
            "Prefect server API incompatible with Python 3.13 (PrefectDBInterface NameError)"
        )

    prev_events = os.environ.get("PREFECT_EVENTS_ENABLED")
    os.environ["PREFECT_EVENTS_ENABLED"] = "false"
    with (
        _temporary_settings(
            {
                _PREFECT_API_URL: None,
                _PREFECT_API_KEY: "testing-disable-events",
            }
        ),
        _quiet_prefect_logging(),
        _prefect_test_harness(),
    ):
        yield
    if prev_events is None:
        os.environ.pop("PREFECT_EVENTS_ENABLED", None)
    else:
        os.environ["PREFECT_EVENTS_ENABLED"] = prev_events
