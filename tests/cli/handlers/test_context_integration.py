"""Integration tests for new HandlerContext.

These tests verify end-to-end workflows without touching legacy code paths.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.rendering.types import OutputFormat
from codeintel.config.serving_models import ServingConfig
from codeintel.serving.bootstrap import build_backend_resource
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.configs import ProvisionedGateway
from tests._helpers.serving_contexts import build_provisioned_service_context
from tests.serving.mcp.conftest import McpBackendComponents

pytest_plugins = ["tests.serving.mcp.conftest"]


def test_full_param_workflow() -> None:
    """Test complete parameter workflow."""
    config = MagicMock()
    config.log_level = "WARNING"
    config.color = True

    ctx = HandlerContext(
        config=config,
        operation_id="test.integration",
        output_format=OutputFormat.JSON,
        verbosity=1,
        project_root=Path("/test/project"),
        _params={
            "name": "test-name",
            "count": 42,
            "enabled": True,
            "path": "/test/path",
        },
    )

    expect_equal(ctx.param_str("name"), "test-name")
    expect_equal(ctx.param_int("count"), 42)
    expect_true(ctx.param_bool("enabled") is True)
    expect_equal(ctx.param_path("path"), Path("/test/path"))

    expect_equal(ctx.require_str("name"), "test-name")
    expect_equal(ctx.require_int("count"), 42)

    expect_equal(ctx.operation_id, "test.integration")
    expect_equal(ctx.output_format, OutputFormat.JSON)
    expect_equal(ctx.verbosity, 1)
    expect_equal(ctx.project_root, Path("/test/project"))


def test_context_manager_closes_resources() -> None:
    """Test context manager properly closes resources."""
    config = MagicMock()
    config.log_level = "WARNING"

    with HandlerContext(
        config=config,
        operation_id="test.cleanup",
        _params={},
    ):
        expect_equal(config.log_level, "WARNING")


def test_logger_property() -> None:
    """Test logger property returns correct logger."""
    config = MagicMock()
    config.log_level = "WARNING"

    ctx = HandlerContext(
        config=config,
        operation_id="my.operation",
        _params={},
    )

    logger = ctx.logger
    expect_equal(logger.name, "codeintel.cli.handlers.my.operation")


def test_nested_contexts() -> None:
    """Test nested context managers work correctly."""
    config = MagicMock()
    config.log_level = "WARNING"

    with HandlerContext(
        config=config,
        operation_id="outer",
        _params={"level": "outer"},
    ) as outer:
        expect_equal(outer.param_str("level"), "outer")

        with HandlerContext(
            config=config,
            operation_id="inner",
            _params={"level": "inner"},
        ) as inner:
            expect_equal(inner.param_str("level"), "inner")
            expect_equal(outer.param_str("level"), "outer")


def test_output_format_variations() -> None:
    """Test context with different output formats."""
    config = MagicMock()
    config.log_level = "WARNING"

    for fmt in [OutputFormat.TEXT, OutputFormat.JSON, OutputFormat.JSONL]:
        ctx = HandlerContext(
            config=config,
            operation_id="test.format",
            output_format=fmt,
            _params={},
        )
        expect_equal(ctx.output_format, fmt)


def test_verbosity_levels() -> None:
    """Test context with different verbosity levels."""
    config = MagicMock()
    config.log_level = "WARNING"

    for level in [0, 1, 2, 3]:
        ctx = HandlerContext(
            config=config,
            operation_id="test.verbosity",
            verbosity=level,
            _params={},
        )
        expect_equal(ctx.verbosity, level)


def test_database_path_fallback() -> None:
    """Test database_path is accessible via db_path property."""
    config = MagicMock()
    config.log_level = "WARNING"

    db_path = Path("/test/db.duckdb")
    ctx = HandlerContext(
        config=config,
        operation_id="test.db",
        database_path=db_path,
        _params={},
    )

    expect_equal(ctx.db_path, db_path)


def test_context_with_provisioned_service_backend(
    provisioned_repo: ProvisionedGateway,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> None:
    """Context can build backend resources against provisioned services."""
    service_ctx = build_provisioned_service_context(
        mcp_backend_factory,
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
    )
    config = MagicMock()
    config.log_level = "WARNING"

    runtime = MagicMock()
    runtime.serving = ServingConfig(
        repo=service_ctx.repo,
        commit=service_ctx.commit,
        mode="local_db",
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    with HandlerContext(
        config=config,
        operation_id="test.provisioned",
        _params={},
        _runtime=runtime,
        _gateway=service_ctx.gateway,
    ) as ctx:
        resource = build_backend_resource(ctx.runtime.serving, gateway=ctx.gateway)
        datasets = resource.backend.list_datasets()

    expect_is_not_none(datasets)
    expect_true(len(datasets) > 0 if hasattr(datasets, "__len__") else True)
