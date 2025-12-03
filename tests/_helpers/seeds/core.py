"""Core seed pack for minimal test data.

This module provides the CorePack which seeds the minimal data needed for
most tests: repo_map, modules, and goids.

The pack provides consistent identifiers and data shapes that other packs
depend on. Most tests should start with CORE_PACK as the foundation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.builders import (
    GoidRow,
    ModuleRow,
    RepoMapRow,
    insert_goids,
    insert_modules,
    insert_repo_map,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Sample Data Constants
# =============================================================================

# Module paths used consistently across tests
MOD_A_PATH = "pkg/mod_a.py"
MOD_B_PATH = "pkg/mod_b.py"
MOD_C_PATH = "pkg/mod_c.py"
MOD_UTIL_PATH = "pkg/util.py"

# Fully qualified module names
MOD_A_FQN = "pkg.mod_a"
MOD_B_FQN = "pkg.mod_b"
MOD_C_FQN = "pkg.mod_c"
MOD_UTIL_FQN = "pkg.util"

# Standard GOID hashes for test functions
# Using consistent hashes enables cross-pack references
GOID_FUNC_A = 1001
GOID_FUNC_B = 1002
GOID_FUNC_C = 1003
GOID_HELPER = 1004
GOID_CALLER = 1005
GOID_CALLEE = 1006


# =============================================================================
# Core Pack Implementation
# =============================================================================


@dataclass
class CorePack:
    """Seed pack for core catalog data.

    Seeds the minimal data needed for most tests:
    - core.repo_map: Repository snapshot metadata
    - core.modules: Module catalog entries
    - core.goids: Global object identifiers for functions

    The seeded data provides a consistent foundation that other packs
    reference via the standard GOID constants.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    module_count : int
        Number of modules to seed.
    function_count : int
        Number of GOIDs (functions) to seed.
    include_util : bool
        Whether to include utility module.
    """

    name: str = "core"
    module_count: int = 3
    function_count: int = 4
    include_util: bool = True
    _dependencies: tuple[SeedPack, ...] = field(default_factory=tuple)

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            Empty tuple; core pack has no dependencies.
        """
        return self._dependencies

    def apply(self, ctx: TestContext) -> None:
        """Apply core seeds to the test context.

        Seeds repo_map, modules, and goids tables with consistent
        test data.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)

        # Seed repo_map
        self._seed_repo_map(ctx, now)

        # Seed modules
        self._seed_modules(ctx)

        # Seed GOIDs
        self._seed_goids(ctx, now)

    def _seed_repo_map(self, ctx: TestContext, now: datetime) -> None:
        """Seed the repo_map table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        modules_dict = {
            MOD_A_FQN: MOD_A_PATH,
            MOD_B_FQN: MOD_B_PATH,
            MOD_C_FQN: MOD_C_PATH,
        }
        if self.include_util:
            modules_dict[MOD_UTIL_FQN] = MOD_UTIL_PATH

        insert_repo_map(
            ctx.gateway,
            [
                RepoMapRow(
                    repo=ctx.repo,
                    commit=ctx.commit,
                    modules=modules_dict,
                    overlays={},
                    generated_at=now,
                ),
            ],
        )

    def _seed_modules(self, ctx: TestContext) -> None:
        """Seed the modules table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        rows = [
            ModuleRow(module=MOD_A_FQN, path=MOD_A_PATH, repo=ctx.repo, commit=ctx.commit),
            ModuleRow(module=MOD_B_FQN, path=MOD_B_PATH, repo=ctx.repo, commit=ctx.commit),
            ModuleRow(module=MOD_C_FQN, path=MOD_C_PATH, repo=ctx.repo, commit=ctx.commit),
        ]
        if self.include_util:
            rows.append(
                ModuleRow(module=MOD_UTIL_FQN, path=MOD_UTIL_PATH, repo=ctx.repo, commit=ctx.commit)
            )

        insert_modules(ctx.gateway, rows[: self.module_count])

    def _seed_goids(self, ctx: TestContext, now: datetime) -> None:
        """Seed the goids table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        goid_rows = [
            GoidRow(
                goid_h128=GOID_FUNC_A,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_A_PATH}#func_a",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_A_PATH,
                kind="function",
                qualname="func_a",
                start_line=1,
                end_line=10,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=GOID_FUNC_B,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_B_PATH}#func_b",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_B_PATH,
                kind="function",
                qualname="func_b",
                start_line=1,
                end_line=15,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=GOID_FUNC_C,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_C_PATH}#func_c",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_C_PATH,
                kind="function",
                qualname="func_c",
                start_line=1,
                end_line=8,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=GOID_HELPER,
                urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_UTIL_PATH}#helper",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_UTIL_PATH,
                kind="function",
                qualname="helper",
                start_line=1,
                end_line=5,
                language="python",
                created_at=now,
            ),
        ]

        insert_goids(ctx.gateway, goid_rows[: self.function_count])


# Default instance for common usage
CORE_PACK = CorePack()


__all__ = [
    "CORE_PACK",
    "GOID_CALLEE",
    "GOID_CALLER",
    "GOID_FUNC_A",
    "GOID_FUNC_B",
    "GOID_FUNC_C",
    "GOID_HELPER",
    "MOD_A_FQN",
    "MOD_A_PATH",
    "MOD_B_FQN",
    "MOD_B_PATH",
    "MOD_C_FQN",
    "MOD_C_PATH",
    "MOD_UTIL_FQN",
    "MOD_UTIL_PATH",
    "CorePack",
]
