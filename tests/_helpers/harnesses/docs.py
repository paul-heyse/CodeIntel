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


@dataclass
class DocsHandlerHarness:
    """Harness providing CommandContexts with runtime stub."""

    ctx: CliTestContext

    @contextmanager
    def command_context(self, params: dict[str, object]) -> Iterator[CommandContext]:
        """Yield a CommandContext ready for docs handlers."""
        with self.ctx.command_context(params) as cmd_ctx:
            cmd_ctx._runtime = _Runtime(  # type: ignore[attr-defined]
                gateway=self.ctx.gateway,
                root=self.ctx.repo_root,
            )
            # Disable storage to skip expensive validation while keeping runtime available.
            cmd_ctx._storage = None  # type: ignore[attr-defined]
            yield cmd_ctx


@contextmanager
def docs_handler_harness(tmp_path: Path) -> Iterator[DocsHandlerHarness]:
    """Create a docs handler harness seeded with core data."""
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK)
    harness = DocsHandlerHarness(ctx=ctx)
    try:
        yield harness
    finally:
        ctx.close()
