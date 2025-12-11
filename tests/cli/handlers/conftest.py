"""Shared fixtures for CLI handler tests.

This module provides fixtures for CLI handler tests following the Testing Charter.
It includes fixtures that provide real gateway implementations via TestContext
and CliTestContext, eliminating the need for mock-based testing.

New tests should prefer:
- `cli_handler_ctx` for simple handler tests
- `graph_cli_ctx` for graph-related handler tests
- `cli_handler_harness_fixture` for harness-based testing
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path

import pytest

from codeintel.cli.context import CommandContext, CommandContextBuilder
from codeintel.storage.gateway import StorageGateway
from tests._helpers.cli_context import CliTestContext, create_cli_test_context
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.harnesses.cli import CliHandlerHarness
from tests._helpers.repo import write_canonical_repo
from tests._helpers.seeds import CORE_PACK, GRAPH_PACK, SUBSYSTEM_PACK
from tests._helpers.serving_contexts import (
    ProvisionedServiceContext,
    build_provisioned_service_context,
)
from tests.serving.mcp.conftest import McpBackendComponents

# Type aliases exported for use by test modules
type CommandContextBuilder_ = Callable[
    [ProvisionedServiceContext, str, dict[str, object]],
    CommandContext,
]

type HandlerContextBuilder = Callable[
    [ProvisionedServiceContext, str, dict[str, object]],
    CommandContext,
]

type CommandContextFactory = Callable[[dict[str, object]], AbstractContextManager[CommandContext]]


@pytest.fixture
def handler_service_context(
    provisioned_repo: TestContext,
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
def handler_context_builder() -> Iterator[HandlerContextBuilder]:
    """Build a CommandContext wired to a provisioned service backend.

    Yields
    ------
    HandlerContextBuilder
        Callable that constructs a CommandContext bound to the given service context.
    """
    stack = ExitStack()

    def _build(
        service_ctx: ProvisionedServiceContext,
        operation_id: str,
        params: dict[str, object],
    ) -> CommandContext:
        # Build CommandContext using the unified builder with injected gateway
        merged_params = dict(params)
        merged_params["_backend_override"] = service_ctx.backend
        builder = (
            CommandContextBuilder()
            .with_params(merged_params)
            .with_operation_id(operation_id)
            .with_injected_gateway(service_ctx.gateway)
        )

        ctx = stack.enter_context(builder.build())
        setattr(ctx.gateway, "backend", service_ctx.backend)
        setattr(ctx, "_backend_override", service_ctx.backend)
        return ctx

    try:
        yield _build
    finally:
        stack.close()


@pytest.fixture
def cli_test_context(tmp_path: Path) -> Iterator[TestContext]:
    """Provision a minimal TestContext with core seeds for CLI handler tests.

    Yields
    ------
    TestContext
        Provisioned context with gateway and core seeds applied.
    """
    ctx = create_test_context(tmp_path)
    write_canonical_repo(ctx.repo_root)
    ctx.require(CORE_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def command_context_factory(cli_test_context: TestContext) -> CommandContextFactory:
    """Build CommandContext instances backed by TestContext gateway.

    Returns
    -------
    CommandContextFactory
        Callable that yields CommandContext objects for given params.
    """

    @contextmanager
    def _build(params: dict[str, object]) -> Iterator[CommandContext]:
        builder = (
            CommandContextBuilder()
            .with_params(params)
            .with_operation_id("cli.test")
            .with_injected_gateway(cli_test_context.gateway)
        )
        with builder.build() as ctx:
            yield ctx

    return _build


# =============================================================================
# New Charter-Compliant Fixtures (Preferred for new tests)
# =============================================================================


@pytest.fixture
def cli_handler_ctx(tmp_path: Path) -> Iterator[CliTestContext]:
    """Provide a CliTestContext with core seeds for handler tests.

    Use this fixture for simple handler tests that need a real gateway
    with core data seeded.

    Yields
    ------
    CliTestContext
        Context with CORE_PACK seeds applied.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def graph_cli_ctx(tmp_path: Path) -> Iterator[CliTestContext]:
    """Provide a CliTestContext with graph seeds for handler tests.

    Use this fixture for graph-related handler tests that need
    call graph, import graph, and related data.

    Yields
    ------
    CliTestContext
        Context with CORE_PACK and GRAPH_PACK seeds applied.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK, GRAPH_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def subsystem_cli_ctx(tmp_path: Path) -> Iterator[CliTestContext]:
    """Provide a CliTestContext with subsystem seeds for handler tests.

    Use this fixture for subsystem-related handler tests.

    Yields
    ------
    CliTestContext
        Context with CORE_PACK and SUBSYSTEM_PACK seeds applied.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK, SUBSYSTEM_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def cli_handler_harness_fixture(tmp_path: Path) -> Iterator[CliHandlerHarness]:
    """Provide a CliHandlerHarness with core seeds.

    Use this fixture when you want the full harness experience with
    execute() method for running handlers.

    Yields
    ------
    CliHandlerHarness
        Harness with CORE_PACK seeds applied.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK)
    harness = CliHandlerHarness(ctx=ctx)
    try:
        yield harness
    finally:
        harness.close()
