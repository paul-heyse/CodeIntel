"""Tests for MCP tool registry.

This module tests tool registration using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.registry import register_tools
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.assertions import expect_equal, expect_not_empty, expect_true
from tests._helpers.gateway import build_duckdb_query_service
from tests._helpers.mcp_registrar import RecordingMcpRegistrar

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway


# =============================================================================
# register_tools Tests
# =============================================================================


def test_register_tools_with_backend(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_tools registers tools on registrar with backend.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service=service,
    )

    registrar = RecordingMcpRegistrar("TestServer")

    # Should not raise
    register_tools(registrar, backend)

    # Server should be configured
    expect_equal(registrar.app_name, "TestServer")
    tools = registrar.list_tools()
    expect_not_empty(tools)
    expect_true(len(tools) >= 6)


def test_register_tools_with_service(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_tools registers tools on registrar with service.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    registrar = RecordingMcpRegistrar("TestServer")

    # Should not raise - accepts QueryService directly
    register_tools(registrar, service)

    expect_equal(registrar.app_name, "TestServer")
    expect_not_empty(registrar.list_tools())


def test_register_tools_with_config(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_tools accepts optional config for auto-pipeline.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    config = ServingConfig(
        mode="local_db",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    registrar = RecordingMcpRegistrar("TestServer")

    # Should not raise
    register_tools(registrar, service, config)

    expect_equal(registrar.app_name, "TestServer")
    expect_not_empty(registrar.list_tools())
