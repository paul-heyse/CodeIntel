"""Entrypoints seed pack for entrypoint/dependency analytics tests.

This module provides the EntrypointsPack which seeds data needed for
testing entrypoint and dependency analytics.

The pack provides modules and GOIDs for a simple pkg.app with hello/cli
entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from textwrap import dedent
from typing import TYPE_CHECKING

from tests._helpers.fixtures.rows import (
    GoidRow,
    ModuleRow,
    dataclass_row,
    insert_rows,
)

if TYPE_CHECKING:
    from pathlib import Path

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
ENTRYPOINTS_HELLO_GOID = 12001
ENTRYPOINTS_CLI_GOID = 12002

# Line numbers for functions
ENTRYPOINTS_HELLO_START = 9
ENTRYPOINTS_HELLO_END = 15
ENTRYPOINTS_CLI_START = 17
ENTRYPOINTS_CLI_END = 22


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
            dataclass_row(
                ModuleRow,
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
            dataclass_row(
                GoidRow,
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
            dataclass_row(
                GoidRow,
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


def entrypoints_source() -> str:
    """Return canonical entrypoints source aligned with pack line numbers.

    Returns
    -------
    str
        Source code for the entrypoints module.
    """
    return dedent(
        """
        from fastapi import FastAPI
        import typer

        app = FastAPI()
        cli = typer.Typer()


        @app.get("/hello")
        def hello(name: str | None = None) -> str:
            \"\"\"Greet the caller.\"\"\"
            greeting = "hello"
            if name:
                greeting = f"hello {name}"
            log_msg = greeting.upper()
            return greeting
        @cli.command()
        def cli_main(count: int = 1) -> str:
            \"\"\"Print greeting count times.\"\"\"
            outputs: list[str] = []
            for _ in range(count):
                outputs.append("hi")
            return ", ".join(outputs)

        if __name__ == "__main__":
            cli()
        """
    ).strip("\n")


def write_entrypoints_source(repo_root: Path) -> str:
    """Write the canonical entrypoints source to the repo root.

    Parameters
    ----------
    repo_root
        Path to the repository root directory.

    Returns
    -------
    str
        Source code that was written to disk.
    """
    source = entrypoints_source()
    path = repo_root / ENTRYPOINTS_MOD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return source


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
    "entrypoints_source",
    "write_entrypoints_source",
]
