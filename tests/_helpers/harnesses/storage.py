"""Harness for storage handler testing with seeded macros/profiles."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from codeintel.storage.gateway import StorageGateway
from tests._helpers.cli_context import CliTestContext, create_cli_test_context
from tests._helpers.seeds import CORE_PACK
from tests._helpers.seeds.cli import STORAGE_PROFILE_PACK


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
    def command_context(self, params: dict[str, object]) -> Iterator[object]:
        """Yield a CommandContext with injected gateway."""
        with self.ctx.command_context(params) as cmd_ctx:
            yield cmd_ctx


@contextmanager
def storage_macro_harness(tmp_path: Path) -> Iterator[StorageHandlerHarness]:
    """Create a storage handler harness seeded with storage profile data."""
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK, STORAGE_PROFILE_PACK)
    # Ensure ingest macros are registered for validation paths
    from codeintel.storage.macros import ensure_ingest_macros  # local import to avoid cyclic init

    ensure_ingest_macros(ctx.gateway.con)
    harness = StorageHandlerHarness(ctx=ctx)
    try:
        yield harness
    finally:
        ctx.close()
