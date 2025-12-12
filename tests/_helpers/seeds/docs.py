"""Seed packs for docs export and MCP backend tests.

This module provides SeedPacks that wrap the existing standalone seed functions
from orchestration/seeding_docs.py, enabling them to be used with TestContext.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.orchestration.seeding_docs import (
    seed_docs_export_minimal,
    seed_mcp_backend,
    seed_profile_data,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


@dataclass(frozen=True)
class DocsExportPack:
    """Seed pack providing minimal data for docs export smoke tests.

    This pack wraps the ``seed_docs_export_minimal`` function, seeding core tables
    (repo_map, modules, goids), graph tables (call_graph, cfg_blocks, import_graph,
    symbol_use), and analytics tables (test_catalog, test_coverage_edges, etc.).
    """

    _name: str = "docs-export"

    @property
    def name(self) -> str:
        """Return unique identifier."""
        return self._name

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed pack dependencies."""
        return ()

    def apply(self, ctx: TestContext) -> None:
        """Apply minimal docs export seeds to the context.

        Delegate to the standalone ``seed_docs_export_minimal`` function for
        actual seeding logic.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        _ = self._name
        seed_docs_export_minimal(
            ctx.gateway,
            repo=ctx.repo,
            commit=ctx.commit,
        )


@dataclass(frozen=True)
class ProfileDataPack:
    """Seed pack providing profile data for analytics tests.

    This pack wraps the ``seed_profile_data`` function.

    Attributes
    ----------
    rel_path : str
        Relative path for seeded data (default: "pkg/mod.py").
    module : str
        Module name for seeded data (default: "pkg.mod").
    """

    _name: str = "profile-data"
    rel_path: str = "pkg/mod.py"
    module: str = "pkg.mod"

    @property
    def name(self) -> str:
        """Return unique identifier."""
        return self._name

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed pack dependencies."""
        return ()

    def apply(self, ctx: TestContext) -> None:
        """Apply profile data seeds to the context.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        seed_profile_data(
            ctx.gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            rel_path=self.rel_path,
            module=self.module,
        )


@dataclass(frozen=True)
class McpBackendPack:
    """Seed pack providing MCP backend test data.

    This pack wraps the ``seed_mcp_backend`` function.
    """

    _name: str = "mcp-backend"

    @property
    def name(self) -> str:
        """Return unique identifier."""
        return self._name

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed pack dependencies."""
        return ()

    def apply(self, ctx: TestContext) -> None:
        """Apply MCP backend seeds to the context.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        _ = self._name
        seed_mcp_backend(
            ctx.gateway,
            repo=ctx.repo,
            commit=ctx.commit,
        )


DOCS_EXPORT_PACK = DocsExportPack()
PROFILE_DATA_PACK = ProfileDataPack()
MCP_BACKEND_PACK = McpBackendPack()


__all__ = [
    "DOCS_EXPORT_PACK",
    "MCP_BACKEND_PACK",
    "PROFILE_DATA_PACK",
    "DocsExportPack",
    "McpBackendPack",
    "ProfileDataPack",
]
