"""Entrypoints seed pack for entrypoint/dependency analytics tests.

This module provides the EntrypointsPack which seeds data needed for
testing entrypoint and dependency analytics.

The pack provides modules and GOIDs for a simple pkg.app with hello/cli
entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.builders import (
    GoidRow,
    ModuleRow,
    insert_rows,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Entrypoints Test Constants
# =============================================================================

# Default repository identifiers
ENTRYPOINTS_REPO = "demo/repo"
ENTRYPOINTS_COMMIT = "deadbeef"

# Module for entrypoints
ENTRYPOINTS_MOD_PATH = "pkg/app.py"
ENTRYPOINTS_MOD_FQN = "pkg.app"

# GOID values for entrypoint functions
ENTRYPOINTS_HELLO_GOID = 1001
ENTRYPOINTS_CLI_GOID = 1002

# Line numbers for functions
ENTRYPOINTS_HELLO_START = 9
ENTRYPOINTS_HELLO_END = 15
ENTRYPOINTS_CLI_START = 17
ENTRYPOINTS_CLI_END = 23


# =============================================================================
# Entrypoints Pack Implementation
# =============================================================================


@dataclass
class EntrypointsPack:
    """Seed pack for entrypoint analytics test data.

    Seeds modules and GOIDs for pkg.app entrypoints including hello and
    cli functions.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    hello_goid : int
        GOID hash for the hello function.
    cli_goid : int
        GOID hash for the cli function.
    repo : str
        Repository identifier.
    commit : str
        Commit identifier.
    """

    name: str = "entrypoints"
    hello_goid: int = ENTRYPOINTS_HELLO_GOID
    cli_goid: int = ENTRYPOINTS_CLI_GOID
    repo: str = ENTRYPOINTS_REPO
    commit: str = ENTRYPOINTS_COMMIT

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            Empty tuple; entrypoints pack has no dependencies.
        """
        return ()

    @property
    def hello_urn(self) -> str:
        """Return URN for hello function.

        Returns
        -------
        str
            URN for the hello function GOID.
        """
        return f"goid:{self.repo}#python:function:{ENTRYPOINTS_MOD_FQN}.hello"

    @property
    def cli_urn(self) -> str:
        """Return URN for cli function.

        Returns
        -------
        str
            URN for the cli function GOID.
        """
        return f"goid:{self.repo}#python:function:{ENTRYPOINTS_MOD_FQN}.cli"

    def apply(self, ctx: TestContext) -> None:
        """Apply entrypoints seeds to the test context.

        Seeds modules and goids for pkg.app entrypoints.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)

        # Seed modules
        self._seed_modules(ctx)

        # Seed GOIDs
        self._seed_goids(ctx, now)

    def _seed_modules(self, ctx: TestContext) -> None:
        """Seed the modules table for entrypoints tests.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            ModuleRow(
                module=ENTRYPOINTS_MOD_FQN,
                path=ENTRYPOINTS_MOD_PATH,
                repo=self.repo,
                commit=self.commit,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_goids(self, ctx: TestContext, now: datetime) -> None:
        """Seed the goids table for entrypoints tests.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        rows = [
            GoidRow(
                goid_h128=self.hello_goid,
                urn=self.hello_urn,
                repo=self.repo,
                commit=self.commit,
                rel_path=ENTRYPOINTS_MOD_PATH,
                kind="function",
                qualname=f"{ENTRYPOINTS_MOD_FQN}.hello",
                start_line=ENTRYPOINTS_HELLO_START,
                end_line=ENTRYPOINTS_HELLO_END,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=self.cli_goid,
                urn=self.cli_urn,
                repo=self.repo,
                commit=self.commit,
                rel_path=ENTRYPOINTS_MOD_PATH,
                kind="function",
                qualname=f"{ENTRYPOINTS_MOD_FQN}.cli",
                start_line=ENTRYPOINTS_CLI_START,
                end_line=ENTRYPOINTS_CLI_END,
                language="python",
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)


# Default instance for common usage
ENTRYPOINTS_PACK = EntrypointsPack()


__all__ = [
    "ENTRYPOINTS_CLI_END",
    "ENTRYPOINTS_CLI_GOID",
    "ENTRYPOINTS_CLI_START",
    "ENTRYPOINTS_COMMIT",
    "ENTRYPOINTS_HELLO_END",
    "ENTRYPOINTS_HELLO_GOID",
    "ENTRYPOINTS_HELLO_START",
    "ENTRYPOINTS_MOD_FQN",
    "ENTRYPOINTS_MOD_PATH",
    "ENTRYPOINTS_PACK",
    "ENTRYPOINTS_REPO",
    "EntrypointsPack",
]
