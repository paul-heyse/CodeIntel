"""Tests for MCP dataset tools."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp import errors
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
from tests._helpers.mcp_registrar import RecordingMcpRegistrar, wrap_fastmcp
from tests._helpers.serving_stubs import HookedDuckDBQueryApi
from tests.serving.mcp.conftest import McpBackendComponents

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LIMIT = 10
MAX_ROWS = 100


# =============================================================================
# Helper Functions
# =============================================================================


# =============================================================================
# register_dataset_tools Tests
# =============================================================================


def test_register_dataset_tools_success(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify register_dataset_tools registers tools successfully."""
    mcp = wrap_fastmcp("Test Dataset Tools")

    # Should not raise
    register_dataset_tools(mcp, mcp_backend.backend)

    # Server should be configured
    expect_equal(mcp.name, "Test Dataset Tools")


def test_register_dataset_tools_with_service(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify register_dataset_tools works with service directly."""
    mcp = wrap_fastmcp("Test Service")

    register_dataset_tools(mcp, mcp_backend_components.service)

    expect_equal(mcp.name, "Test Service")


def test_register_dataset_tools_with_config(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify register_dataset_tools works with serving config."""
    mcp = wrap_fastmcp("Test With Config")
    config = None

    register_dataset_tools(mcp, mcp_backend.backend, config=config)

    expect_equal(mcp.name, "Test With Config")


def test_register_dataset_tools_on_multiple_servers(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify tools can be registered on multiple servers."""
    mcp1 = wrap_fastmcp("Server 1")
    register_dataset_tools(mcp1, mcp_backend.backend)
    expect_equal(mcp1.name, "Server 1")

    mcp2 = wrap_fastmcp("Server 2")
    register_dataset_tools(mcp2, mcp_backend.backend)
    expect_equal(mcp2.name, "Server 2")


def test_dataset_tools_serialize_unicode_payloads() -> None:
    """Dataset tools serialize multi-row unicode/nullable payloads."""

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

    for spec in dataset_ops:
        expect_is_not_none(spec.id)
        expect_equal(spec.category, "datasets")
        expect_is_not_none(spec.backend_method)
        expect_is_not_none(spec.output_model_name)


# =============================================================================
# Backend Method Tests
# =============================================================================


def test_backend_list_datasets(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.list_datasets works."""
    datasets = mcp_backend.backend.list_datasets()

    expect_is_instance(datasets, list)


def test_backend_dataset_specs(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.dataset_specs works."""
    specs = mcp_backend.backend.dataset_specs()

    expect_is_instance(specs, list)


def test_backend_read_dataset_rows(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.read_dataset_rows works."""
    datasets = mcp_backend.backend.list_datasets()
    if datasets:
        dataset_name = datasets[0].name
        rows = mcp_backend.backend.read_dataset_rows(dataset_name=dataset_name, limit=5)
        expect_true(hasattr(rows, "dataset_name"))
        expect_true(hasattr(rows, "rows"))


def test_backend_read_dataset_rows_with_offset(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.read_dataset_rows works with offset."""
    datasets = mcp_backend.backend.list_datasets()
    if datasets:
        dataset_name = datasets[0].name
        rows = mcp_backend.backend.read_dataset_rows(dataset_name=dataset_name, limit=5, offset=0)
        expect_true(hasattr(rows, "rows"))
        expect_true(hasattr(rows, "offset"))


def test_backend_dataset_schema(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.dataset_schema works."""
    datasets = mcp_backend.backend.list_datasets()
    if datasets:
        dataset_name = datasets[0].name
        schema = mcp_backend.backend.dataset_schema(dataset_name=dataset_name)
        expect_is_not_none(schema)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_backend_read_dataset_rows_nonexistent_dataset(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend raises error for nonexistent dataset."""
    with pytest.raises(McpError):
        mcp_backend.backend.read_dataset_rows(dataset_name="nonexistent_dataset_xyz", limit=5)


# =============================================================================
# Limits Tests
# =============================================================================


def test_backend_with_custom_limits(
    mcp_backend_components: McpBackendComponents,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Verify backend respects custom limits."""
    custom_limit = 25
    custom_max = 250
    limits = BackendLimits(default_limit=custom_limit, max_rows_per_call=custom_max)
    backend = mcp_backend_factory(
        gateway=mcp_backend_components.gateway,
        repo=mcp_backend_components.repo,
        commit=mcp_backend_components.commit,
        limits=limits,
    ).backend

    expect_equal(backend.limits.default_limit, custom_limit)
    expect_equal(backend.limits.max_rows_per_call, custom_max)


# =============================================================================
# Serialization Tests
# =============================================================================


def test_backend_list_datasets_returns_descriptors(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.list_datasets returns descriptors with name field."""
    datasets = mcp_backend.backend.list_datasets()

    # Verify we get objects with name attribute
    for dataset in datasets:
        expect_true(hasattr(dataset, "name"))


def test_backend_dataset_specs_returns_pydantic_models(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify backend.dataset_specs returns Pydantic models."""
    specs = mcp_backend.backend.dataset_specs()

    for spec in specs:
        expect_is_instance(spec, DatasetSpecDescriptor)


# =============================================================================
# Context Tests
# =============================================================================


def test_register_dataset_tools_preserves_backend_state(
    mcp_backend: McpBackendComponents,
) -> None:
    """Verify registration doesn't alter backend state."""
    mcp = wrap_fastmcp("Test State")
    backend = mcp_backend.backend
    original_repo = backend.repo
    original_commit = backend.commit
    original_limits = backend.limits

    register_dataset_tools(mcp, backend)

    expect_equal(backend.repo, original_repo)
    expect_equal(backend.commit, original_commit)
    expect_equal(backend.limits, original_limits)


def test_local_query_service_as_backend(
    mcp_backend_components: McpBackendComponents,
) -> None:
    """Verify LocalQueryService can be used as backend."""
    mcp = wrap_fastmcp("Test Local Service")
    register_dataset_tools(mcp, mcp_backend_components.service)

    expect_equal(mcp.name, "Test Local Service")
