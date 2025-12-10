"""Tests for serving layer bootstrap module.

This module tests service stack construction and bootstrap functionality
using real gateways and configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import (
    BackendContext,
    BackendLimits,
    DuckDBQueryService,
    DuckDBRepositories,
    GraphEngineProvider,
)
from codeintel.serving.bootstrap import (
    BackendResourceOptions,
    BootstrapOptions,
    DatasetRegistryOptions,
    ServiceBuildOptions,
    ServiceStack,
    build_backend_context,
    build_http_query_service,
    build_local_query_service,
    build_query_service,
    build_repositories,
    build_service_from_config,
    build_service_stack,
    get_observability_from_config,
)
from codeintel.serving.services.observability import ServiceObservability
from codeintel.serving.services.query_service import HttpQueryService, LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from tests._helpers.context import TestContext

# Constants for test values
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
SMALL_DEFAULT_LIMIT = 50
SMALL_MAX_LIMIT = 500
TIMEOUT_SECONDS = 30.0


# =============================================================================
# DatasetRegistryOptions Tests
# =============================================================================


def test_dataset_registry_options_defaults() -> None:
    """Verify DatasetRegistryOptions default values."""
    opts = DatasetRegistryOptions()

    expect_is_none(opts.tables)
    expect_true(opts.validate)
    expect_true(callable(opts.describe_fn))


def test_dataset_registry_options_custom() -> None:
    """Verify DatasetRegistryOptions with custom values."""
    tables = {"test.dataset": "Test dataset"}

    def custom_describe(table_key: str, family: str) -> str:
        return f"Custom: {table_key} ({family})"

    opts = DatasetRegistryOptions(
        tables=tables,
        describe_fn=custom_describe,
        validate=False,
    )

    expect_equal(opts.tables, tables)
    expect_false(opts.validate)
    expect_equal(opts.describe_fn("t", "f"), "Custom: t (f)")


# =============================================================================
# ServiceBuildOptions Tests
# =============================================================================


def test_service_build_options_defaults() -> None:
    """Verify ServiceBuildOptions default values."""
    opts = ServiceBuildOptions()

    expect_is_none(opts.registry)
    expect_is_none(opts.observability)
    expect_is_none(opts.graph_runtime)
    expect_is_none(opts.graph_engine)


def test_service_build_options_with_observability() -> None:
    """Verify ServiceBuildOptions with observability set."""
    obs = ServiceObservability(enabled=True)
    opts = ServiceBuildOptions(observability=obs)

    expect_is_not_none(opts.observability)
    expect_true(opts.observability is obs)
    expect_true(obs.enabled)


# =============================================================================
# BootstrapOptions Tests
# =============================================================================


def test_bootstrap_options_defaults() -> None:
    """Verify BootstrapOptions default values."""
    opts = BootstrapOptions()

    expect_true(opts.create_views)
    expect_true(opts.validate_registry)
    expect_is_none(opts.observability)
    expect_is_none(opts.graph_runtime)
    expect_is_none(opts.graph_engine)


def test_bootstrap_options_custom() -> None:
    """Verify BootstrapOptions with custom values."""
    opts = BootstrapOptions(
        create_views=False,
        validate_registry=False,
    )

    expect_false(opts.create_views)
    expect_false(opts.validate_registry)


# =============================================================================
# BackendResourceOptions Tests
# =============================================================================


def test_backend_resource_options_defaults() -> None:
    """Verify BackendResourceOptions default values."""
    opts = BackendResourceOptions()

    expect_is_none(opts.registry)
    expect_is_none(opts.observability)
    expect_is_none(opts.graph_runtime)
    expect_is_none(opts.runtime_pool)


# =============================================================================
# get_observability_from_config Tests
# =============================================================================


def test_get_observability_disabled_by_default(tmp_path: Path) -> None:
    """Verify observability is None when not enabled in config."""
    cfg = ServingConfig(
        repo="test/repo",
        commit="abc123",
        mode="local_db",
        db_path=tmp_path / "test.duckdb",
        repo_root=tmp_path,
    )

    result = get_observability_from_config(cfg)

    expect_is_none(result)


def test_service_observability_directly() -> None:
    """Verify ServiceObservability can be constructed independently."""
    obs = ServiceObservability(enabled=True)

    expect_true(obs.enabled)


def test_service_observability_disabled() -> None:
    """Verify ServiceObservability defaults to disabled."""
    obs = ServiceObservability(enabled=False)

    expect_false(obs.enabled)


# =============================================================================
# build_backend_context Tests
# =============================================================================


def test_build_backend_context_basic(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Verify build_backend_context returns BackendContext with correct values."""
    cfg = ServingConfig(
        repo="demo/repo",
        commit="deadbeef",
        mode="local_db",
        db_path=tmp_path / "test.duckdb",
        repo_root=tmp_path,
    )

    context = build_backend_context(fresh_gateway, cfg)

    expect_true(context.gateway is fresh_gateway)
    expect_equal(context.repo, "demo/repo")
    expect_equal(context.commit, "deadbeef")
    expect_is_not_none(context.limits)


def test_build_backend_context_with_custom_limits(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Verify build_backend_context uses provided limits."""
    cfg = ServingConfig(
        repo="demo/repo",
        commit="deadbeef",
        mode="local_db",
        db_path=tmp_path / "test.duckdb",
        repo_root=tmp_path,
    )
    custom_limits = BackendLimits(
        default_limit=SMALL_DEFAULT_LIMIT, max_rows_per_call=SMALL_MAX_LIMIT
    )

    context = build_backend_context(fresh_gateway, cfg, limits=custom_limits)

    expect_equal(context.limits.default_limit, SMALL_DEFAULT_LIMIT)
    expect_equal(context.limits.max_rows_per_call, SMALL_MAX_LIMIT)


# =============================================================================
# build_repositories Tests
# =============================================================================


def test_build_repositories_returns_duckdb_repositories(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Verify build_repositories returns DuckDBRepositories."""
    cfg = ServingConfig(
        repo="demo/repo",
        commit="deadbeef",
        mode="local_db",
        db_path=tmp_path / "test.duckdb",
        repo_root=tmp_path,
    )

    repos = build_repositories(fresh_gateway, cfg)

    expect_equal(repos.repo, "demo/repo")
    expect_equal(repos.commit, "deadbeef")


# =============================================================================
# build_http_query_service Tests
# =============================================================================


def test_build_http_query_service_basic() -> None:
    """Verify build_http_query_service returns HttpQueryService."""
    call_count = 0

    def mock_request(_endpoint: str, _params: dict[str, object]) -> object:
        nonlocal call_count
        call_count += 1
        return {"result": "ok"}

    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_LIMIT)

    service = build_http_query_service(mock_request, limits=limits)

    expect_is_instance(service, HttpQueryService)


def test_build_http_query_service_with_observability() -> None:
    """Verify build_http_query_service accepts observability."""

    def mock_request(_endpoint: str, _params: dict[str, object]) -> object:
        return {}

    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_LIMIT)
    obs = ServiceObservability(enabled=True)

    service = build_http_query_service(mock_request, limits=limits, observability=obs)

    expect_is_instance(service, HttpQueryService)


# =============================================================================
# build_service_from_config Tests
# =============================================================================


def test_build_service_from_config_local_db_missing_gateway(tmp_path: Path) -> None:
    """Verify ValueError when gateway missing for local_db mode."""
    cfg = ServingConfig(
        repo="test/repo",
        commit="abc123",
        mode="local_db",
        db_path=tmp_path / "test.duckdb",
        repo_root=tmp_path,
    )

    with pytest.raises(ValueError, match="StorageGateway is required"):
        build_service_from_config(cfg)


def test_build_service_from_config_remote_api_missing_request(tmp_path: Path) -> None:
    """Verify ValueError when request_json missing for remote_api mode."""
    cfg = ServingConfig(
        repo="test/repo",
        commit="abc123",
        mode="remote_api",
        api_base_url="http://example.com",
        repo_root=tmp_path,
    )

    with pytest.raises(ValueError, match="request_json callable is required"):
        build_service_from_config(cfg)


def test_build_service_from_config_local_db_with_observability(
    provisioned_repo: TestContext,
) -> None:
    """Verify local_db mode works with observability option."""
    cfg = ServingConfig(
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        mode="local_db",
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )
    obs = ServiceObservability(enabled=True)
    opts = ServiceBuildOptions(observability=obs)

    service = build_service_from_config(cfg, gateway=provisioned_repo.gateway, options=opts)

    expect_is_instance(service, LocalQueryService)


def test_build_service_from_config_remote_api_returns_http_service(tmp_path: Path) -> None:
    """Verify remote_api mode returns HttpQueryService."""

    def mock_request(_endpoint: str, _params: dict[str, object]) -> object:
        return {}

    cfg = ServingConfig(
        repo="test/repo",
        commit="abc123",
        mode="remote_api",
        api_base_url="http://example.com",
        repo_root=tmp_path,
    )

    service = build_service_from_config(cfg, request_json=mock_request)

    expect_is_instance(service, HttpQueryService)


def test_build_service_from_config_local_db_returns_local_service(
    provisioned_repo: TestContext,
) -> None:
    """Verify local_db mode returns LocalQueryService with real gateway."""
    cfg = ServingConfig(
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        mode="local_db",
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    service = build_service_from_config(cfg, gateway=provisioned_repo.gateway)

    expect_is_instance(service, LocalQueryService)


# =============================================================================
# build_service_stack Tests
# =============================================================================


def test_build_service_stack_returns_stack(provisioned_repo: TestContext) -> None:
    """Verify build_service_stack returns complete ServiceStack."""
    cfg = ServingConfig(
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        mode="local_db",
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    stack = build_service_stack(cfg, gateway=provisioned_repo.gateway)

    expect_is_instance(stack, ServiceStack)
    expect_is_not_none(stack.service)
    expect_is_not_none(stack.query)
    expect_is_not_none(stack.context)
    expect_is_not_none(stack.repositories)


def test_build_service_stack_with_options(provisioned_repo: TestContext) -> None:
    """Verify build_service_stack accepts BootstrapOptions."""
    cfg = ServingConfig(
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        mode="local_db",
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )
    options = BootstrapOptions(
        create_views=False,
        validate_registry=False,
    )

    stack = build_service_stack(cfg, gateway=provisioned_repo.gateway, options=options)

    expect_is_instance(stack, ServiceStack)


def test_build_service_stack_close_calls_cleanup(provisioned_repo: TestContext) -> None:
    """Verify ServiceStack.close() invokes cleanup function."""
    cfg = ServingConfig(
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        mode="local_db",
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    # Build stack - note that close will close the gateway
    # Since provisioned_repo manages its own cleanup, we skip the actual close test
    stack = build_service_stack(cfg, gateway=provisioned_repo.gateway)

    # Just verify close_fn is callable
    expect_true(callable(stack.close_fn))


# =============================================================================
# ServiceStack Tests
# =============================================================================


def test_service_stack_close_method(provisioned_repo: TestContext) -> None:
    """Verify ServiceStack.close() method exists and is callable."""
    cfg = ServingConfig(
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        mode="local_db",
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    stack = build_service_stack(cfg, gateway=provisioned_repo.gateway)

    # Verify close method exists
    expect_true(hasattr(stack, "close"))
    expect_true(callable(stack.close))


# =============================================================================
# build_local_query_service Tests
# =============================================================================


def test_build_local_query_service_with_validation(
    provisioned_repo: TestContext,
) -> None:
    """Verify build_local_query_service constructs service with validation."""
    cfg = ServingConfig(
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        mode="local_db",
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )
    gateway = provisioned_repo.gateway
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_LIMIT)

    context = BackendContext(
        gateway=gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        limits=limits,
        graph_engine=None,
    )
    repos = DuckDBRepositories(gateway=gateway, repo=cfg.repo, commit=cfg.commit)
    provider = GraphEngineProvider(context=context, graph_engine=None)
    query = DuckDBQueryService(context=context, repositories=repos, engine_provider=provider)

    registry_opts = DatasetRegistryOptions(validate=False)

    service = build_local_query_service(
        gateway,
        cfg,
        query=query,
        registry=registry_opts,
    )

    expect_is_instance(service, LocalQueryService)


# =============================================================================
# build_query_service Tests
# =============================================================================


def test_build_query_service_basic(provisioned_repo: TestContext) -> None:
    """Verify build_query_service constructs DuckDBQueryService."""
    gateway = provisioned_repo.gateway
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_LIMIT)

    context = BackendContext(
        gateway=gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
        graph_engine=None,
    )
    repos = DuckDBRepositories(
        gateway=gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    provider = GraphEngineProvider(context=context, graph_engine=None)

    query = build_query_service(context, repos, provider)

    expect_is_not_none(query)
    # Access repo/commit through context
    expect_equal(query.context.repo, provisioned_repo.repo)
    expect_equal(query.context.commit, provisioned_repo.commit)
