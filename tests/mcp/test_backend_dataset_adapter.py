"""Dataset delegation tests for MCP backend adapters."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import (
    FunctionValidationRow,
    RepoMapRow,
)
from tests._helpers.row_protocol import insert_rows
from tests._helpers.gateway import build_duckdb_backend


@pytest.fixture
def gateway(fresh_gateway: StorageGateway) -> StorageGateway:
    """
    Real gateway with repo identity seeded for adapter tests.

    Returns
    -------
    StorageGateway
        Gateway configured with repo/commit identity.
    """
    insert_rows(
        fresh_gateway,
        [
            RepoMapRow(
                repo="r",
                commit="c",
                modules={},
                overlays={},
            )
        ],
    )
    return fresh_gateway


def test_read_dataset_rows_delegates(gateway: StorageGateway) -> None:
    """Adapters should delegate dataset reads directly to the service."""
    backend = build_duckdb_backend(gateway, repo="r", commit="c")
    insert_rows(
        gateway,
        [
            FunctionValidationRow(
                repo="r",
                commit="c",
                function_goid_h128=1,
                rel_path="pkg/a.py",
                qualname="pkg.a.fn",
                issue="ok",
                detail="",
                created_at=datetime.now(UTC),
            )
        ],
    )
    resp = backend.read_dataset_rows(dataset_name="function_validation", limit=100, offset=0)
    if not resp.rows:
        pytest.fail("Expected rows from stub service")
    if resp.limit is None or resp.limit <= 0:
        pytest.fail(f"Unexpected limit {resp.limit}")
