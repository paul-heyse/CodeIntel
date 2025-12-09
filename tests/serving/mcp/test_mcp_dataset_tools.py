"""Tests for MCP dataset tools.

This module tests the dataset browsing MCP tools registered from Operation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.dataset_tools import DatasetToolOptions, register_dataset_tools
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.mcp.models import DatasetSpecDescriptor
from codeintel.serving.operations import iter_operations
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.assertions import (
    assert_logged,
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.dataset_factories import make_descriptor
from tests._helpers.gateway import build_duckdb_query_service
from tests._helpers.mcp_registrar import RecordingMcpRegistrar, wrap_fastmcp
from tests._helpers.serving_stubs import HookedDuckDBQueryApi

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LIMIT = 10
MAX_ROWS = 100


# =============================================================================
# Helper Functions
# =============================================================================


def _build_backend(provisioned_repo: ProvisionedGateway) -> DuckDBBackend:
    """Build a DuckDBBackend for testing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.

    Returns
    -------
    DuckDBBackend
        Configured backend.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    return DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
        observability=None,
        service=service,
    )


# =============================================================================
# register_dataset_tools Tests
# =============================================================================


def test_register_dataset_tools_success(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_dataset_tools registers tools successfully.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Dataset Tools")
    backend = _build_backend(provisioned_repo)

    # Should not raise
    register_dataset_tools(mcp, backend)

    # Server should be configured
    expect_equal(mcp.name, "Test Dataset Tools")


def test_register_dataset_tools_with_service(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_dataset_tools works with service directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Service")
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    register_dataset_tools(mcp, service)

    expect_equal(mcp.name, "Test Service")


def test_register_dataset_tools_with_config(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_dataset_tools works with serving config.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test With Config")
    backend = _build_backend(provisioned_repo)
    config = ServingConfig()

    register_dataset_tools(mcp, backend, config=config)

    expect_equal(mcp.name, "Test With Config")


def test_register_dataset_tools_on_multiple_servers(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify tools can be registered on multiple servers.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    mcp1 = wrap_fastmcp("Server 1")
    register_dataset_tools(mcp1, backend)
    expect_equal(mcp1.name, "Server 1")

    mcp2 = wrap_fastmcp("Server 2")
    register_dataset_tools(mcp2, backend)
    expect_equal(mcp2.name, "Server 2")


def test_dataset_tools_serialize_unicode_payloads() -> None:
    """Dataset tools serialize multi-row unicode/nullable payloads."""
    payload = [
        make_descriptor(
            name="datasets.alpha",
            table="core.alpha",
            description="データセット alpha",
        )[0],
        make_descriptor(
            name="datasets.delta",
            table="docs.δelta",
            description="Docs delta",
            options=None,
        )[0],
    ]

    class _ListDatasetService(LocalQueryService):
        def __init__(self, dataset_payload: Sequence[dm.DatasetDescriptorDomain]) -> None:
            super().__init__(query=HookedDuckDBQueryApi())
            self._dataset_payload = dataset_payload

        def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
            return self._call("list_datasets", lambda: list(self._dataset_payload))

    backend = _ListDatasetService(payload)

    registrar = RecordingMcpRegistrar("dataset-recorder")
    ops = [spec for spec in iter_operations() if spec.id == "datasets.list"]
    register_dataset_tools(
        registrar,
        backend,
        options=DatasetToolOptions(operations=ops),
    )

    tool = registrar.registry["list_datasets"]
    result = cast("list[dict[str, object]]", tool())
    expect_is_instance(result, list)
    expect_equal(len(result), 2)
    expect_true(any(row["owner"] is None for row in result))
    expect_true(
        any(
            ("δ" in str(row.get("table", ""))) or ("δ" in str(row.get("jsonl_filename") or ""))
            for row in result
        )
    )


def test_dataset_tools_log_problem_detail(caplog: pytest.LogCaptureFixture) -> None:
    """Dataset tools emit ProblemDetail payloads and warning logs on failure."""

    class _FailingService(LocalQueryService):
        def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
            self.calls.append("list_datasets")
            message = "list failed"
            raise errors.backend_failure(message)

    failing_backend = _FailingService(query=HookedDuckDBQueryApi())

    registrar = RecordingMcpRegistrar("dataset-errors")
    ops = [spec for spec in iter_operations() if spec.id == "datasets.list"]
    register_dataset_tools(
        registrar,
        failing_backend,
        options=DatasetToolOptions(operations=ops),
    )

    with caplog.at_level("WARNING"):
        result_dict: dict[str, object] = cast(
            "dict[str, object]", registrar.registry["list_datasets"]()
        )

    expect_true("error" in result_dict)
    error_payload = cast("dict[str, object]", result_dict["error"])
    expect_equal(error_payload["title"], "Backend failure")
    assert_logged(caplog.records, level="WARNING", containing="MCP tool error: list failed")


# =============================================================================
# Operation Tests
# =============================================================================


def test_iter_operations_yields_dataset_operations() -> None:
    """Verify iter_operations yields dataset category operations."""
    dataset_ops = [spec for spec in iter_operations() if spec.category == "datasets"]

    expect_true(len(dataset_ops) > 0)
    # Dataset operations should have tool_name
    ops_with_tools = [op for op in dataset_ops if op.tool_name is not None]
    expect_true(len(ops_with_tools) > 0)


def test_dataset_operations_have_required_fields() -> None:
    """Verify dataset operations have required fields."""
    dataset_ops = [spec for spec in iter_operations() if spec.category == "datasets"]

    for spec in dataset_ops:
        expect_is_not_none(spec.id)
        expect_equal(spec.category, "datasets")
        expect_is_not_none(spec.backend_method)
        expect_is_not_none(spec.output_model_name)


# =============================================================================
# Backend Method Tests
# =============================================================================


def test_backend_list_datasets(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.list_datasets works.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    datasets = backend.list_datasets()

    expect_is_instance(datasets, list)


def test_backend_dataset_specs(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.dataset_specs works.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    specs = backend.dataset_specs()

    expect_is_instance(specs, list)


def test_backend_read_dataset_rows(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.read_dataset_rows works.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    datasets = backend.list_datasets()
    if datasets:
        dataset_name = datasets[0].name
        rows = backend.read_dataset_rows(dataset_name=dataset_name, limit=5)
        expect_true(hasattr(rows, "dataset_name"))
        expect_true(hasattr(rows, "rows"))


def test_backend_read_dataset_rows_with_offset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.read_dataset_rows works with offset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    datasets = backend.list_datasets()
    if datasets:
        dataset_name = datasets[0].name
        rows = backend.read_dataset_rows(dataset_name=dataset_name, limit=5, offset=0)
        expect_true(hasattr(rows, "rows"))
        expect_true(hasattr(rows, "offset"))


def test_backend_dataset_schema(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.dataset_schema works.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    datasets = backend.list_datasets()
    if datasets:
        dataset_name = datasets[0].name
        schema = backend.dataset_schema(dataset_name=dataset_name)
        expect_is_not_none(schema)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_backend_read_dataset_rows_nonexistent_dataset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend raises error for nonexistent dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    with pytest.raises(McpError):
        backend.read_dataset_rows(dataset_name="nonexistent_dataset_xyz", limit=5)


# =============================================================================
# Limits Tests
# =============================================================================


def test_backend_with_custom_limits(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend respects custom limits.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    custom_limit = 25
    custom_max = 250
    limits = BackendLimits(default_limit=custom_limit, max_rows_per_call=custom_max)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
        observability=None,
        service=service,
    )

    expect_equal(backend.limits.default_limit, custom_limit)
    expect_equal(backend.limits.max_rows_per_call, custom_max)


# =============================================================================
# Serialization Tests
# =============================================================================


def test_backend_list_datasets_returns_descriptors(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.list_datasets returns descriptors with name field.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    datasets = backend.list_datasets()

    # Verify we get objects with name attribute
    for dataset in datasets:
        expect_true(hasattr(dataset, "name"))


def test_backend_dataset_specs_returns_pydantic_models(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.dataset_specs returns Pydantic models.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    specs = backend.dataset_specs()

    for spec in specs:
        expect_is_instance(spec, DatasetSpecDescriptor)


# =============================================================================
# Context Tests
# =============================================================================


def test_register_dataset_tools_preserves_backend_state(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify registration doesn't alter backend state.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test State")
    backend = _build_backend(provisioned_repo)

    original_repo = backend.repo
    original_commit = backend.commit
    original_limits = backend.limits

    register_dataset_tools(mcp, backend)

    expect_equal(backend.repo, original_repo)
    expect_equal(backend.commit, original_commit)
    expect_equal(backend.limits, original_limits)


def test_local_query_service_as_backend(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService can be used as backend.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    mcp = wrap_fastmcp("Test Local Service")
    register_dataset_tools(mcp, service)

    expect_equal(mcp.name, "Test Local Service")
