"""Dataset delegation tests for MCP backend adapters."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from codeintel.mcp.backend import DuckDBBackend
from codeintel.mcp.models import DatasetRowsResponse, ResponseMeta, ViewRow
from codeintel.services.query_service import LocalQueryService


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


def _fake_gateway() -> object:
    """
    Minimal gateway stub to satisfy DuckDBBackend signature.

    Returns
    -------
    object
        Stub object with gateway attributes.
    """
    return SimpleNamespace(
        con=None,
        config=SimpleNamespace(repo="r", commit="c", read_only=True),
        datasets=SimpleNamespace(mapping={}),
        close=lambda: None,
    )


def test_read_dataset_rows_delegates() -> None:
    """Adapters should delegate dataset reads directly to the service."""
    backend = DuckDBBackend(
        gateway=_fake_gateway(),  # type: ignore[arg-type]
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
