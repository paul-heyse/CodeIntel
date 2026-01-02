"""Pipeline seed pack for pipeline integration tests.

This module provides the PipelinePack which seeds data needed for
pipeline graph integration tests.

The pack provides consistent module and GOID data for testing the full
pipeline from graph building through analytics validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.assertions import ModulesAssertions
from tests._helpers.fixtures.rows import (
    GoidRow,
    ModuleRow,
    RepoMapRow,
    TestCatalogRow,
    dataclass_row,
    insert_rows,
)
from tests._helpers.modules_expectations import modules_expected_from_repo_tree
from tests._helpers.seeds.span import (
    SPAN_CALLER_END,
    SPAN_CALLER_GOID,
    SPAN_CALLER_START,
    SPAN_COMMIT,
    SPAN_MOD_A_FQN,
    SPAN_MOD_A_PATH,
    SPAN_MOD_B_FQN,
    SPAN_MOD_B_PATH,
    SPAN_REPO,
    SPAN_TEST_ID,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.context import SeedPack, TestContext


PIPELINE_REPO = SPAN_REPO
PIPELINE_COMMIT = SPAN_COMMIT


PIPELINE_CALLEE_GOID = 100
PIPELINE_CALLEE_START = 1
PIPELINE_CALLEE_END = 2


@dataclass
class PipelinePack:
    """Seed pack for pipeline integration test data.

    Seeds modules, GOIDs (caller and callee), and test catalog entries
    for pipeline tests. This pack provides both functions in the call chain.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    callee_goid : int
        GOID hash for the callee function.
    caller_goid : int
        GOID hash for the caller function.
    """

    name: str = "pipeline"
    callee_goid: int = PIPELINE_CALLEE_GOID
    caller_goid: int = SPAN_CALLER_GOID

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            Empty tuple; pipeline pack has no dependencies.
        """
        return ()

    def apply(self, ctx: TestContext) -> None:
        """Apply pipeline seeds to the test context.

        Seeds modules, goids (callee and caller), and test catalog.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)

        module_map = self._resolve_module_map(ctx)
        self._seed_repo_map(ctx, module_map)
        self._seed_modules(ctx, module_map)
        ModulesAssertions(ctx.gateway, ctx.snapshot).inventory_consistent()

        self._seed_goids(ctx, now)

        self._seed_test_catalog(ctx, now)

    @staticmethod
    def _seed_repo_map(ctx: TestContext, module_map: dict[str, str]) -> None:
        """Seed the core.repo_map table."""
        rows = [
            dataclass_row(
                RepoMapRow,
                repo=PIPELINE_REPO,
                commit=PIPELINE_COMMIT,
                modules=module_map,
                overlays={},
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_modules(ctx: TestContext, module_map: dict[str, str]) -> None:
        """Seed the modules table for pipeline tests.

        Parameters
        ----------
        ctx
            Test context with gateway.
        module_map
            Module map keyed by module name to repo-relative paths.
        """
        rows = [
            dataclass_row(
                ModuleRow, module=module, path=path, repo=PIPELINE_REPO, commit=PIPELINE_COMMIT
            )
            for module, path in sorted(module_map.items())
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _resolve_module_map(ctx: TestContext) -> dict[str, str]:
        path_map = modules_expected_from_repo_tree(ctx.repo_root)
        module_map = {module: path for path, module in path_map.items()}
        if not module_map:
            module_map = {
                SPAN_MOD_A_FQN: SPAN_MOD_A_PATH,
                SPAN_MOD_B_FQN: SPAN_MOD_B_PATH,
            }
        return module_map

    def _seed_goids(self, ctx: TestContext, now: datetime) -> None:
        """Seed the goids table for pipeline tests.

        Seeds both callee and caller functions.

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
                goid_h128=self.callee_goid,
                urn=f"urn:{SPAN_MOD_A_FQN}.callee",
                repo=PIPELINE_REPO,
                commit=PIPELINE_COMMIT,
                rel_path=SPAN_MOD_A_PATH,
                kind="function",
                qualname=f"{SPAN_MOD_A_FQN}.callee",
                start_line=PIPELINE_CALLEE_START,
                end_line=PIPELINE_CALLEE_END,
                language="python",
                created_at=now,
            ),
            dataclass_row(
                GoidRow,
                goid_h128=self.caller_goid,
                urn=f"urn:{SPAN_MOD_B_FQN}.caller",
                repo=PIPELINE_REPO,
                commit=PIPELINE_COMMIT,
                rel_path=SPAN_MOD_B_PATH,
                kind="function",
                qualname=f"{SPAN_MOD_B_FQN}.caller",
                start_line=SPAN_CALLER_START,
                end_line=SPAN_CALLER_END,
                language="python",
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_test_catalog(ctx: TestContext, now: datetime) -> None:
        """Seed the test catalog for pipeline tests.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        rows = [
            dataclass_row(
                TestCatalogRow,
                test_id=SPAN_TEST_ID,
                repo=PIPELINE_REPO,
                commit=PIPELINE_COMMIT,
                rel_path=SPAN_MOD_B_PATH,
                qualname=f"{SPAN_MOD_B_FQN}.caller",
                status="passed",
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)


PIPELINE_PACK = PipelinePack()


def write_pipeline_repo_files(repo_root: Path) -> tuple[int, int]:
    """Write repository files for pipeline integration tests.

    Creates a minimal package structure with caller/callee functions
    for testing the full pipeline.

    Parameters
    ----------
    repo_root
        Root directory for the test repository.

    Returns
    -------
    tuple[int, int]
        Start and end line numbers of the caller function.
    """
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")
    (pkg_dir / "a.py").write_text("def callee():\n    return 1\n", encoding="utf8")
    (pkg_dir / "b.py").write_text(
        "from pkg.a import callee\n\ndef caller():\n    return callee()\n",
        encoding="utf8",
    )
    return SPAN_CALLER_START, SPAN_CALLER_END


__all__ = [
    "PIPELINE_CALLEE_END",
    "PIPELINE_CALLEE_GOID",
    "PIPELINE_CALLEE_START",
    "PIPELINE_COMMIT",
    "PIPELINE_PACK",
    "PIPELINE_REPO",
    "PipelinePack",
    "write_pipeline_repo_files",
]
