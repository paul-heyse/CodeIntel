"""CLI test context for handler testing.

This module provides CliTestContext which wraps TestContext with
CommandContext building capabilities. It enables tests to execute
CLI handlers with real gateways and proper resource lifecycle management.

The design follows the hexagonal architecture pattern used by analytics
plugin tests, ensuring consistent testing patterns across the codebase.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from codeintel.cli.context import CommandContextBuilder
from tests._helpers.context import create_test_context
from tests._helpers.repo import write_canonical_repo

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.cli.context import CommandContext
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.context import SeedPack, TestContext


@dataclass
class CliTestContext:
    """Test context wrapper with CommandContext building capabilities.

    Provide a unified interface for CLI handler tests that combines
    TestContext functionality with CommandContext construction.

    Attributes
    ----------
    test_ctx : TestContext
        Underlying test context with gateway and snapshot.
    operation_id : str
        Default operation ID for command contexts.
    _exit_stack : ExitStack
        Stack for managing context lifecycles.

    Examples
    --------
    >>> from tests._helpers.cli_context import create_cli_test_context
    >>> from tests._helpers.seeds import CORE_PACK
    >>>
    >>> def test_handler(tmp_path):  # doctest: +SKIP
    ...     ctx = create_cli_test_context(tmp_path)
    ...     ctx.require(CORE_PACK)
    ...     with ctx.command_context({"name": "test"}) as cmd_ctx:
    ...         result = my_handler(cmd_ctx)
    ...     ctx.close()
    """

    test_ctx: TestContext
    operation_id: str = "cli.test"
    _exit_stack: ExitStack = field(default_factory=ExitStack)

    @property
    def gateway(self) -> StorageGateway:
        """Return the storage gateway.

        Returns
        -------
        StorageGateway
            Gateway from the underlying test context.
        """
        return self.test_ctx.gateway

    @property
    def repo(self) -> str:
        """Return repository identifier.

        Returns
        -------
        str
            Repository slug from snapshot.
        """
        return self.test_ctx.repo

    @property
    def commit(self) -> str:
        """Return commit identifier.

        Returns
        -------
        str
            Commit hash from snapshot.
        """
        return self.test_ctx.commit

    @property
    def repo_root(self) -> Path:
        """Return repository root path.

        Returns
        -------
        Path
            Path to repository root.
        """
        return self.test_ctx.repo_root

    @property
    def build_dir(self) -> Path:
        """Return build directory path."""
        return self.test_ctx.build_dir

    def require(self, *seed_packs: SeedPack) -> Self:
        """Ensure seed packs are applied (idempotent).

        Delegate to the underlying TestContext.

        Parameters
        ----------
        seed_packs
            One or more seed packs to apply.

        Returns
        -------
        Self
            Self for method chaining.
        """
        self.test_ctx.require(*seed_packs)
        return self

    @contextmanager
    def command_context(
        self,
        params: dict[str, object] | None = None,
        *,
        operation_id: str | None = None,
    ) -> Iterator[CommandContext]:
        """Build a CommandContext for handler execution.

        Create a CommandContext with the test gateway injected and
        provided parameters.

        Parameters
        ----------
        params
            Handler parameters dictionary.
        operation_id
            Override operation ID (uses default if not provided).

        Yields
        ------
        CommandContext
            Configured command context for handler execution.

        Examples
        --------
        >>> with ctx.command_context({"key": "value"}) as cmd_ctx:  # doctest: +SKIP
        ...     result = handler(cmd_ctx)
        """
        from codeintel.cli.context import CommandContextBuilder

        builder = (
            CommandContextBuilder()
            .with_params(params or {})
            .with_operation_id(operation_id or self.operation_id)
            .with_injected_gateway(self.test_ctx.gateway)
        )
        with builder.build() as ctx:
            yield ctx

    def build_command_context(
        self,
        params: dict[str, object] | None = None,
        *,
        operation_id: str | None = None,
    ) -> CommandContext:
        """Build and enter a CommandContext, tracking it for cleanup.

        Use this when you need the context outside a with block.
        The context will be cleaned up when close() is called.

        Parameters
        ----------
        params
            Handler parameters dictionary.
        operation_id
            Override operation ID (uses default if not provided).

        Returns
        -------
        CommandContext
            Configured command context for handler execution.
        """
        from codeintel.cli.context import CommandContextBuilder

        builder = (
            CommandContextBuilder()
            .with_params(params or {})
            .with_operation_id(operation_id or self.operation_id)
            .with_injected_gateway(self.test_ctx.gateway)
        )
        return self._exit_stack.enter_context(builder.build())

    def query_count(self, table: str, where: str | None = None) -> int:
        """Count rows in a table.

        Delegate to TestContext.query_count for consistency.

        Parameters
        ----------
        table
            Table name (schema.table format).
        where
            Optional WHERE clause (without 'WHERE' keyword).

        Returns
        -------
        int
            Number of rows.
        """
        return self.test_ctx.query_count(table, where)

    def close(self) -> None:
        """Close the underlying context and any managed resources."""
        self._exit_stack.close()
        self.test_ctx.close()

    def __enter__(self) -> Self:
        """Enter context manager scope.

        Returns
        -------
        Self
            Self reference for use in with block.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager and close resources."""
        self.close()


def create_cli_test_context(
    tmp_path: Path,
    *,
    operation_id: str = "cli.test",
    write_repo: bool = True,
) -> CliTestContext:
    """Create a CliTestContext for handler testing.

    Factory function that sets up a minimal test environment with
    gateway and optional repository files.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    operation_id
        Default operation ID for command contexts.
    write_repo
        Whether to write canonical repository files.

    Returns
    -------
    CliTestContext
        Configured CLI test context.

    Examples
    --------
    >>> from tests._helpers.cli_context import create_cli_test_context
    >>>
    >>> def test_example(tmp_path):  # doctest: +SKIP
    ...     ctx = create_cli_test_context(tmp_path)
    ...     ctx.require(CORE_PACK)
    ...     # ... use ctx ...
    ...     ctx.close()
    """
    test_ctx = create_test_context(tmp_path)
    if write_repo:
        write_canonical_repo(test_ctx.repo_root)
    return CliTestContext(test_ctx=test_ctx, operation_id=operation_id)


def cli_test_context_with_seeds(
    tmp_path: Path,
    *seed_packs: SeedPack,
    operation_id: str = "cli.test",
) -> CliTestContext:
    """Create a CliTestContext with specified seed packs applied.

    Convenience factory that creates the context and applies seed packs
    in one step.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    seed_packs
        Seed packs to apply.
    operation_id
        Default operation ID for command contexts.

    Returns
    -------
    CliTestContext
        Context with seeds applied.

    Examples
    --------
    >>> from tests._helpers.cli_context import cli_test_context_with_seeds
    >>> from tests._helpers.seeds import CORE_PACK, GRAPH_PACK
    >>>
    >>> def test_graph_handler(tmp_path):  # doctest: +SKIP
    ...     with cli_test_context_with_seeds(tmp_path, CORE_PACK, GRAPH_PACK) as ctx:
    ...         # ... use ctx ...
    """
    ctx = create_cli_test_context(tmp_path, operation_id=operation_id)
    if seed_packs:
        ctx.require(*seed_packs)
    return ctx


@contextmanager
def make_command_context(
    params: dict[str, object] | None = None,
    *,
    operation_id: str = "test.op",
) -> Iterator[CommandContext]:
    """Create a standalone CommandContext for testing.

    Use this for simple handler tests that don't require seeded data
    or gateway access. For tests needing a real gateway, use CliTestContext.

    Parameters
    ----------
    params
        Handler parameters dictionary.
    operation_id
        Operation identifier for tracing.

    Yields
    ------
    CommandContext
        Configured command context.

    Examples
    --------
    >>> with make_command_context({"name": "test"}) as ctx:  # doctest: +SKIP
    ...     result = my_handler(ctx)
    """
    builder = CommandContextBuilder().with_params(params or {}).with_operation_id(operation_id)
    with builder.build() as ctx:
        yield ctx


def params(**kwargs: object) -> dict[str, object]:
    """Create a type-safe params dictionary for CommandContext.

    This helper ensures the dict type is `dict[str, object]` to avoid
    type variance issues when passing to CommandContextBuilder.with_params().

    Parameters
    ----------
    **kwargs
        Parameter key-value pairs.

    Returns
    -------
    dict[str, object]
        Type-safe params dictionary.

    Examples
    --------
    >>> p = params(name="test", count=42, flag=True)
    >>> type(p)
    <class 'dict'>
    """
    return dict(kwargs)


__all__ = [
    "CliTestContext",
    "cli_test_context_with_seeds",
    "create_cli_test_context",
    "make_command_context",
    "params",
]
