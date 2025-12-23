"""Span seed pack for graph span alignment tests.

This module provides the SpanPack which seeds data needed for span
alignment tests across call graph, CFG/DFG, and coverage components.

The pack provides consistent module and GOID data for validating that
different graph components agree on function spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.assertions import ModulesAssertions
from tests._helpers.builders import (
    GoidRow,
    ModuleRow,
    RepoMapRow,
    TestCatalogRow,
    insert_rows,
)
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.context import SeedPack, TestContext


SPAN_REPO = "demo/repo"
SPAN_COMMIT = "deadbeef"


SPAN_MOD_A_PATH = "pkg/a.py"
SPAN_MOD_B_PATH = "pkg/b.py"
SPAN_MOD_A_FQN = "pkg.a"
SPAN_MOD_B_FQN = "pkg.b"


SPAN_CALLER_GOID = 200
SPAN_CALLER_START = 3
SPAN_CALLER_END = 4


SPAN_TEST_ID = "tests/test_sample.py::test_caller"


@dataclass
class SpanPack:
    """Seed pack for span alignment test data.

    Seeds modules, GOIDs, and test catalog entries for span alignment
    tests. This pack provides the database rows needed to test that
    call graph, CFG/DFG, and coverage components agree on function spans.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    caller_goid : int
        GOID hash for the caller function.
    caller_start : int
        Start line of caller function.
    caller_end : int
        End line of caller function.
    """

    name: str = "span"
    caller_goid: int = SPAN_CALLER_GOID
    caller_start: int = SPAN_CALLER_START
    caller_end: int = SPAN_CALLER_END

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            Empty tuple; span pack has no dependencies.
        """
        return ()

    def apply(self, ctx: TestContext) -> None:
        """Apply span seeds to the test context.

        Seeds modules, goids, and test catalog for span alignment tests.

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
            RepoMapRow(
                repo=SPAN_REPO,
                commit=SPAN_COMMIT,
                modules=module_map,
                overlays={},
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_modules(ctx: TestContext, module_map: dict[str, str]) -> None:
        """Seed the modules table for span tests.

        Parameters
        ----------
        ctx
            Test context with gateway.
        module_map
            Module map keyed by module name to repo-relative paths.
        """
        rows = [
            ModuleRow(module=module, path=path, repo=SPAN_REPO, commit=SPAN_COMMIT)
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
        """Seed the goids table for span tests.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        rows = [
            GoidRow(
                goid_h128=self.caller_goid,
                urn=f"urn:{SPAN_MOD_B_FQN}.caller",
                repo=SPAN_REPO,
                commit=SPAN_COMMIT,
                rel_path=SPAN_MOD_B_PATH,
                kind="function",
                qualname=f"{SPAN_MOD_B_FQN}.caller",
                start_line=self.caller_start,
                end_line=self.caller_end,
                language="python",
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_test_catalog(ctx: TestContext, now: datetime) -> None:
        """Seed the test catalog for span tests.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        rows = [
            TestCatalogRow(
                test_id=SPAN_TEST_ID,
                repo=SPAN_REPO,
                commit=SPAN_COMMIT,
                rel_path=SPAN_MOD_B_PATH,
                qualname=f"{SPAN_MOD_B_FQN}.caller",
                status="passed",
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)


SPAN_PACK = SpanPack()


def write_span_repo_files(repo_root: Path) -> tuple[int, int]:
    """Write repository files for span alignment tests.

    Creates a minimal package structure with caller/callee functions
    for testing span alignment across graph components.

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
    "SPAN_CALLER_END",
    "SPAN_CALLER_GOID",
    "SPAN_CALLER_START",
    "SPAN_COMMIT",
    "SPAN_MOD_A_FQN",
    "SPAN_MOD_A_PATH",
    "SPAN_MOD_B_FQN",
    "SPAN_MOD_B_PATH",
    "SPAN_PACK",
    "SPAN_REPO",
    "SPAN_TEST_ID",
    "SpanPack",
    "write_span_repo_files",
]
