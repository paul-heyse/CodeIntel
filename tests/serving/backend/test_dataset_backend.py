"""Tests for DatasetBackend behavior."""

from __future__ import annotations

import pytest

from codeintel.serving.backend import BackendContext, BackendLimits, DuckDBRepositories
from codeintel.serving.backend.dataset_backend import DatasetBackend
from codeintel.serving.mcp import errors
from codeintel.storage.gateway import StorageGateway


def _expect(*, condition: bool, message: str) -> None:
    """Fail the test when a condition is not met."""
    if not condition:
        pytest.fail(message)


def _build_components(
    gateway: StorageGateway, limits: BackendLimits | None = None
) -> tuple[BackendContext, DuckDBRepositories]:
    repo = gateway.config.repo or "demo/repo"
    commit = gateway.config.commit or "deadbeef"
    context = BackendContext(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=limits or BackendLimits(),
        graph_engine=None,
    )
    repositories = DuckDBRepositories(gateway, context.repo, context.commit)
    return context, repositories


def test_dataset_schema_includes_columns(architecture_gateway: StorageGateway) -> None:
    """Return schema details with DuckDB columns populated."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetBackend(context=context, repositories=repositories)

    schema = backend.dataset_schema(dataset_name="call_graph_edges")

    _expect(
        condition=schema.dataset_name == "call_graph_edges",
        message="Dataset name should round-trip",
    )
    _expect(condition=bool(schema.duckdb_schema), message="DuckDB schema should not be empty")
    _expect(condition=schema.meta is not None, message="Response metadata should be populated")


def test_read_dataset_rows_invalid_offset(architecture_gateway: StorageGateway) -> None:
    """Raise invalid-argument when offset is negative."""
    context, repositories = _build_components(architecture_gateway)
    backend = DatasetBackend(context=context, repositories=repositories)

    with pytest.raises(errors.McpError) as excinfo:
        backend.read_dataset_rows(dataset_name="call_graph_edges", limit=5, offset=-1)

    _expect(
        condition=excinfo.value.detail.code == "invalid-argument",
        message="Negative offsets should raise invalid-argument",
    )
