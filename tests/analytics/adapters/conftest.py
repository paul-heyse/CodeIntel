"""Shared fixtures for analytics adapter tests.

This module provides common fixtures for adapter tests to reduce duplication
across the adapter test suite. All fixtures use production-grade components
following the Testing Charter principles.
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
# Gateway Fixtures
# =============================================================================


@pytest.fixture
def adapter_gateway() -> Iterator[StorageGateway]:
    """Provide in-memory gateway with full schema for adapter tests.

    This fixture creates a production-parity gateway with all schemas
    and macros applied, suitable for testing adapter behavior.

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
# Snapshot Fixtures
# =============================================================================


@pytest.fixture
def adapter_snapshot() -> SnapshotRef:
    """Provide standard snapshot reference for adapter tests.

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
