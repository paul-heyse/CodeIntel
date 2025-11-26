"""Dataset delegation tests for MCP backend adapters."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import (
    FunctionValidationRow,
    RepoMapRow,
    insert_function_validation,
    insert_repo_map,
)


@pytest.fixture
def gateway(fresh_gateway: StorageGateway) -> StorageGateway:
    """
    Real gateway with repo identity seeded for adapter tests.

    Returns
    -------
    StorageGateway
        Gateway configured with repo/commit identity.
    """
    insert_repo_map(
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
    backend = DuckDBBackend(
        gateway=gateway,  # type: ignore[arg-type]
        repo="r",
        commit="c",
        service_override=None,
    )
    insert_function_validation(
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
