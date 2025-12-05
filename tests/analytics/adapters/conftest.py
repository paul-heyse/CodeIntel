"""Shared fixtures for analytics adapter tests.

This module provides adapter-specific fixtures. For general test fixtures like
TestContext, test_ctx, graph_ctx, etc., use the fixtures from the main conftest.py.

Most adapter tests should use the standard test_ctx fixture from conftest.py.
The adapter_gateway and adapter_snapshot fixtures are provided for backward
compatibility with existing tests.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from codeintel.storage.macros import ensure_ingest_macros
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.fakes import create_test_snapshot

# =============================================================================
# Constants (aliases for backward compatibility)
# =============================================================================

ADAPTER_TEST_REPO = DEFAULT_REPO
ADAPTER_TEST_COMMIT = DEFAULT_COMMIT


# =============================================================================
# Legacy Gateway Fixtures (prefer test_ctx from main conftest)
# =============================================================================


@pytest.fixture
def adapter_gateway() -> Iterator[StorageGateway]:
    """Provide in-memory gateway with full schema for adapter tests.

    .. note::
        Prefer using the test_ctx fixture from main conftest.py instead.
        This fixture is provided for backward compatibility.

    Yields
    ------
    StorageGateway
        Gateway with schema and macros applied.
    """
    gw = open_memory_gateway(
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
    )
    apply_all_schemas(gw.con)
    ensure_ingest_macros(gw.con)
    try:
        yield gw
    finally:
        gw.close()


# =============================================================================
# Legacy Snapshot Fixtures (prefer test_ctx.to_snapshot_ref() instead)
# =============================================================================


@pytest.fixture
def adapter_snapshot() -> SnapshotRef:
    """Provide standard snapshot reference for adapter tests.

    .. note::
        Prefer using test_ctx.to_snapshot_ref() instead.
        This fixture is provided for backward compatibility.

    Returns
    -------
    SnapshotRef
        Snapshot with test repo and commit identifiers.
    """
    return create_test_snapshot()


__all__ = [
    "ADAPTER_TEST_COMMIT",
    "ADAPTER_TEST_REPO",
    "adapter_gateway",
    "adapter_snapshot",
]
