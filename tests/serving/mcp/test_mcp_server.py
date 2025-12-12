"""Tests for MCP server creation and configuration.

This module tests the MCP server factory using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.bootstrap import BackendResource
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.server import create_mcp_server
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.assertions import expect_true
from tests._helpers.gateway import build_duckdb_query_service
from tests._helpers.mcp_registrar import wrap_fastmcp

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


def test_create_mcp_server_local_db_requires_gateway() -> None:
    """Verify create_mcp_server raises ValueError without gateway in local_db mode."""
    cfg = ServingConfig(
        mode="local_db",
        repo="test/repo",
        commit="abc123",
    )

    with pytest.raises(ValueError, match="StorageGateway is required"):
        create_mcp_server(cfg, gateway=None)


def test_create_mcp_server_returns_server_and_close(
    provisioned_repo: TestContext,
) -> None:
    """Verify create_mcp_server returns an MCP server and close callback.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    cfg = ServingConfig(
        mode="local_db",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    server, close = create_mcp_server(
        cfg, gateway=provisioned_repo.gateway, mcp_factory=wrap_fastmcp
    )

    expect_true(hasattr(server, "run"))
    expect_true(callable(close))


def test_create_mcp_server_with_custom_backend_factory(
    provisioned_repo: TestContext,
) -> None:
    """Verify create_mcp_server accepts custom backend factory.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    factory_called = False

    def custom_factory(
        _cfg: ServingConfig,
        **_kwargs: object,
    ) -> BackendResource:
        nonlocal factory_called
        factory_called = True
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
        return BackendResource(backend=backend, service=service, close=lambda: None)

    cfg = ServingConfig(
        mode="local_db",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    server, _close = create_mcp_server(
        cfg,
        backend_factory=custom_factory,
        gateway=provisioned_repo.gateway,
        mcp_factory=wrap_fastmcp,
    )

    expect_true(factory_called)
    expect_true(hasattr(server, "run"))


def test_create_mcp_server_with_custom_tools_registration(
    provisioned_repo: TestContext,
) -> None:
    """Verify create_mcp_server accepts custom tool registration function.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    tools_registered = False

    def custom_register_tools(_server: object, _backend: object, _cfg: object) -> None:
        nonlocal tools_registered
        tools_registered = True

    cfg = ServingConfig(
        mode="local_db",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    server, _close = create_mcp_server(
        cfg,
        gateway=provisioned_repo.gateway,
        register_tools_fn=custom_register_tools,
        mcp_factory=wrap_fastmcp,
    )

    expect_true(tools_registered)
    expect_true(hasattr(server, "run"))


def test_create_mcp_server_close_callback_callable(
    provisioned_repo: TestContext,
) -> None:
    """Verify the close callback can be invoked without error.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    close_called = False

    def track_close() -> None:
        nonlocal close_called
        close_called = True

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

    def custom_factory(
        _cfg: ServingConfig,
        **_kwargs: object,
    ) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=track_close)

    cfg = ServingConfig(
        mode="local_db",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    _server, close = create_mcp_server(
        cfg,
        backend_factory=custom_factory,
        gateway=provisioned_repo.gateway,
    )

    close()
    expect_true(close_called)


def test_create_mcp_server_default_tools_registered(
    provisioned_repo: TestContext,
) -> None:
    """Verify create_mcp_server registers default tools when no custom fn provided.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    cfg = ServingConfig(
        mode="local_db",
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    server, _close = create_mcp_server(
        cfg, gateway=provisioned_repo.gateway, mcp_factory=wrap_fastmcp
    )

    expect_true(hasattr(server, "tools"))
