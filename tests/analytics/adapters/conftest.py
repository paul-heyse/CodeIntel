"""Shared fixtures for analytics adapter tests.

This module provides adapter-specific fixtures. For general test fixtures like
TestContext, test_ctx, graph_ctx, etc., use the fixtures from the main conftest.py.

Most adapter tests should use the standard test_ctx fixture from conftest.py.
The adapter_gateway and adapter_snapshot fixtures are provided for backward
compatibility with existing tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from codeintel.storage.ingest_macros import ensure_ingest_macros
from codeintel.storage.schemas import apply_all_schemas

# =============================================================================
# Constants
# =============================================================================

ADAPTER_TEST_REPO = "test/repo"
ADAPTER_TEST_COMMIT = "abc123"


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
    return SnapshotRef(
        repo=ADAPTER_TEST_REPO,
        commit=ADAPTER_TEST_COMMIT,
        repo_root=Path.cwd(),
    )


__all__ = [
    "ADAPTER_TEST_COMMIT",
    "ADAPTER_TEST_REPO",
    "adapter_gateway",
    "adapter_snapshot",
]
