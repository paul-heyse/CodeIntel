"""Dataset delegation tests for MCP backend adapters."""

from __future__ import annotations

import pytest

from codeintel.mcp.backend import DuckDBBackend
from codeintel.mcp.models import DatasetRowsResponse, ResponseMeta, ViewRow
from codeintel.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import RepoMapRow, insert_repo_map


class DatasetStubService(LocalQueryService):
    """Stub service that returns deterministic dataset rows."""

    def __init__(self) -> None:
        super().__init__(query=None, dataset_tables={})  # type: ignore[arg-type]

    def read_dataset_rows(  # noqa: PLR6301
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:  # pragma: no cover
        """
        Return a fixed dataset response for testing.

        Returns
        -------
        DatasetRowsResponse
            Dataset rows without adapter-side clamping.
        """
        _ = (limit, offset)
        return DatasetRowsResponse(
            dataset=dataset_name,
            limit=5,
            offset=offset,
            rows=[ViewRow.model_validate({"issue": "ok"})],
            meta=ResponseMeta(),
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


def test_read_dataset_rows_delegates(gateway: object) -> None:
    """Adapters should delegate dataset reads directly to the service."""
    backend = DuckDBBackend(
        gateway=gateway,  # type: ignore[arg-type]
        repo="r",
        commit="c",
        service_override=DatasetStubService(),
    )
    resp = backend.read_dataset_rows(dataset_name="function_validation", limit=100, offset=0)
    if not resp.rows:
        pytest.fail("Expected rows from stub service")
    expected_limit = 5
    if resp.limit != expected_limit:
        pytest.fail(f"Expected stub limit to be preserved, got {resp.limit}")
