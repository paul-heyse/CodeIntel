"""Shared fixtures for CLI handler tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from codeintel.cli.handlers.context import HandlerContext
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageGateway
from tests._helpers.configs import ProvisionedGateway
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.serving_contexts import (
    ProvisionedServiceContext,
    build_provisioned_service_context,
)
from tests.serving.mcp.conftest import McpBackendComponents

pytest_plugins = ["tests.serving.mcp.conftest"]

type HandlerContextBuilder = Callable[
    [ProvisionedServiceContext, str, dict[str, object]],
    HandlerContext,
]


@pytest.fixture
def handler_service_context(
    provisioned_repo: ProvisionedGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> ProvisionedServiceContext:
    """Provisioned LocalQueryService/Backend for handler tests.

    Returns
    -------
    ProvisionedServiceContext
        Context built from the ingested repository snapshot.
    """
    return build_provisioned_service_context(
        mcp_backend_factory,
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
    )


@pytest.fixture
def architecture_service_context(
    architecture_gateway: StorageGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> ProvisionedServiceContext:
    """Provisioned context seeded with architecture data.

    Returns
    -------
    ProvisionedServiceContext
        Context backed by architecture gateway seeds.
    """
    return build_provisioned_service_context(
        mcp_backend_factory,
        gateway=architecture_gateway,
        snapshot=(DEFAULT_REPO, DEFAULT_COMMIT),
    )


@pytest.fixture
def handler_context_builder() -> HandlerContextBuilder:
    """Build a HandlerContext wired to a provisioned service backend.

    Returns
    -------
    HandlerContextBuilder
        Callable that constructs a HandlerContext bound to the given service context.
    """

    def _build(
        service_ctx: ProvisionedServiceContext,
        operation_id: str,
        params: dict[str, object],
    ) -> HandlerContext:
        gateway_config = getattr(service_ctx.gateway, "config", None)
        repo_root = getattr(gateway_config, "repo_root", Path.cwd())
        db_path = getattr(gateway_config, "db_path", None)
        serving = ServingConfig(
            mode="local_db",
            repo_root=repo_root,
            repo=service_ctx.repo,
            commit=service_ctx.commit,
            db_path=db_path,
            default_limit=service_ctx.limits.default_limit,
            max_rows_per_call=service_ctx.limits.max_rows_per_call,
        )

        runtime = MagicMock()
        runtime.serving = serving
        runtime.paths = MagicMock()
        runtime.paths.db_path = db_path
        runtime.repo = service_ctx.repo
        runtime.commit = service_ctx.commit

        runtime_backend = SimpleNamespace(backend="duckdb", use_gpu=False)
        runtime_options = SimpleNamespace(features=None, snapshot=None, cache_key=None)
        graph_runtime = SimpleNamespace(
            options=runtime_options,
            engine=None,
            backend=runtime_backend,
        )
        return HandlerContext(
            config=MagicMock(),
            operation_id=operation_id,
            _params=params,
            _runtime=runtime,
            _gateway=service_ctx.gateway,
            _graph_runtime=graph_runtime,
        )

    return _build
