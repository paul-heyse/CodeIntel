"""Harness for docs handler testing without mocks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from codeintel.cli.context import CommandContext
from tests._helpers.cli_context import CliTestContext, create_cli_test_context
from tests._helpers.seeds import CORE_PACK


@dataclass
class _Runtime:
    """Minimal runtime stub with gateway and root."""

    gateway: object
    root: Path

    @property
    def runtime(self) -> _Runtime:
        """Return self to satisfy CommandContext.runtime accessor."""
        return self


@dataclass
class DocsHandlerHarness:
    """Harness providing CommandContexts with runtime stub."""

    ctx: CliTestContext

    @contextmanager
    def command_context(self, params: dict[str, object]) -> Iterator[CommandContext]:
        """Yield a CommandContext ready for docs handlers.

        Yields
        ------
        CommandContext
            Context with runtime stub and storage disabled.
        """
        with self.ctx.command_context(params) as cmd_ctx:
            cmd_ctx.__dict__["_runtime"] = _Runtime(
                gateway=self.ctx.gateway,
                root=self.ctx.repo_root,
            )
            # Disable storage to skip expensive validation while keeping runtime available.
            cmd_ctx.__dict__["_storage"] = None
            yield cmd_ctx


@contextmanager
def docs_handler_harness(tmp_path: Path) -> Iterator[DocsHandlerHarness]:
    """Create a docs handler harness seeded with core data.

    Yields
    ------
    DocsHandlerHarness
        Harness exposing docs-specific command contexts.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK)
    harness = DocsHandlerHarness(ctx=ctx)
    try:
        yield harness
    finally:
        ctx.close()
