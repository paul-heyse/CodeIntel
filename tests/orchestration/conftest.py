"""Prefect test fixtures for quiet, graceful orchestration runs."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from prefect.server.api import server as prefect_server
from prefect.testing.utilities import prefect_test_harness


@contextmanager
def _quiet_prefect_logging() -> Iterator[None]:
    """Suppress Prefect/Rich console logging to avoid noisy teardown errors."""
    logging.disable(logging.CRITICAL)
    prefect_logger = logging.getLogger("prefect")
    subprocess_logger = prefect_server.subprocess_server_logger

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
    """
    Run Prefect flows against the ephemeral test harness with minimal logging.

    The harness starts a temporary API/database and ensures clean shutdown,
    preventing teardown errors or CRASHED states when the process exits.
    """
    with (
        _quiet_prefect_logging(),
        prefect_test_harness(),
    ):
        yield
