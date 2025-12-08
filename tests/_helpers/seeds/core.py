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

from tests._helpers.builders import GoidRow, ModuleRow, RepoMapRow, insert_rows
from tests._helpers.repo import (
    GOID_CALLEE,
    GOID_CALLER,
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_FQN,
    MOD_A_PATH,
    MOD_B_FQN,
    MOD_B_PATH,
    MOD_C_FQN,
    MOD_C_PATH,
    MOD_UTIL_FQN,
    MOD_UTIL_PATH,
    CanonicalRepo,
    write_canonical_repo,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Sample Data Constants
# =============================================================================

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
        canonical = write_canonical_repo(ctx.repo_root)

        # Seed repo_map
        self._seed_repo_map(ctx, now, canonical)

        # Seed modules
        self._seed_modules(ctx, canonical)

        # Seed GOIDs
        self._seed_goids(ctx, now, canonical)

    def _seed_repo_map(self, ctx: TestContext, now: datetime, canonical: CanonicalRepo) -> None:
        """Seed the repo_map table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        canonical
            Canonical repo metadata.
        """
        modules_dict = dict(canonical.module_paths)
        if not self.include_util and MOD_UTIL_FQN in modules_dict:
            modules_dict.pop(MOD_UTIL_FQN, None)

        insert_rows(
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

    def _seed_modules(self, ctx: TestContext, canonical: CanonicalRepo) -> None:
        """Seed the modules table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        canonical
            Canonical repo metadata.
        """
        rows = [
            ModuleRow(module=module, path=path, repo=ctx.repo, commit=ctx.commit)
            for module, path in canonical.module_paths.items()
            if self.include_util or module != MOD_UTIL_FQN
        ]

        insert_rows(ctx.gateway, rows[: self.module_count])

    def _seed_goids(self, ctx: TestContext, now: datetime, canonical: CanonicalRepo) -> None:
        """Seed the goids table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        canonical
            Canonical repo metadata.
        """
        goid_rows = []
        for qualname in ("func_a", "func_b", "func_c", "helper"):
            if not self.include_util and qualname == "helper":
                continue
            meta = canonical.functions[qualname]
            goid_rows.append(
                GoidRow(
                    goid_h128=meta.goid,
                    urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{meta.rel_path}#{meta.qualname}",
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path=meta.rel_path,
                    kind="function",
                    qualname=meta.qualname,
                    start_line=meta.start_line,
                    end_line=meta.end_line,
                    language="python",
                    created_at=now,
                )
            )

        insert_rows(ctx.gateway, goid_rows[: self.function_count])


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
