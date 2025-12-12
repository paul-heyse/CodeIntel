"""Harness for storage handler testing with seeded macros/profiles."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.macros import ensure_ingest_macros
from tests._helpers.cli_context import create_cli_test_context
from tests._helpers.env_options import EnvOptions
from tests._helpers.seeds import CORE_PACK
from tests._helpers.seeds.cli import STORAGE_PROFILE_PACK

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.cli.context import CommandContext
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.cli_context import CliTestContext


@dataclass
class StorageHandlerHarness:
    """Harness providing a seeded gateway and CommandContext builder."""

    ctx: CliTestContext

    @property
    def gateway(self) -> StorageGateway:
        return self.ctx.gateway

    @property
    def db_path(self) -> Path:
        return self.ctx.build_dir / "db" / "codeintel.duckdb"

    @contextmanager
    def command_context(self, params: dict[str, object]) -> Iterator[CommandContext]:
        """Yield a CommandContext with injected gateway.

        Yields
        ------
        CommandContext
            CommandContext bound to the seeded gateway.
        """
        with self.ctx.command_context(params) as cmd_ctx:
            yield cmd_ctx


@contextmanager
def storage_macro_harness(tmp_path: Path) -> Iterator[StorageHandlerHarness]:
    """Create a storage handler harness seeded with storage profile data.

    Yields
    ------
    StorageHandlerHarness
        Harness with gateway prepared for storage handlers.
    """
    ctx = create_cli_test_context(tmp_path, options=EnvOptions(file_backed=True))
    ctx.require(CORE_PACK, STORAGE_PROFILE_PACK)

    ensure_ingest_macros(ctx.gateway.con)
    harness = StorageHandlerHarness(ctx=ctx)
    try:
        yield harness
    finally:
        ctx.close()
