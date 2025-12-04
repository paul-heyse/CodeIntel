"""Validation seed pack for graph validation tests.

This module provides the ValidationPack which seeds data that triggers
graph validation warnings, useful for testing validation rule detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.builders import GoidRow, ModuleRow, insert_rows

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Validation Data Constants
# =============================================================================

REPO: str = "demo/repo"
COMMIT: str = "deadbeef"
CALLER_GOID: int = 1
ORPHAN_GOID: int = 999


# =============================================================================
# Validation Pack Implementation
# =============================================================================


@dataclass
class ValidationPack:
    """Seed pack for graph validation gap detection.

    Seeds data that intentionally creates validation gaps:
    - AST nodes without corresponding GOIDs
    - Modules for validation context
    - GOIDs with references to missing dependencies

    This pack is useful for testing that validation rules correctly
    detect inconsistencies in the graph data.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    """

    name: str = "validation"
    _dependencies: tuple[SeedPack, ...] = field(default_factory=tuple)

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            No dependencies - this pack is self-contained.
        """
        return ()

    def apply(self, ctx: TestContext) -> None:
        """Apply validation gap seeds to the test context.

        Seeds data that triggers graph validation warnings.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)
        self._seed_ast_nodes(ctx)
        self._seed_modules(ctx)
        self._seed_goids(ctx, now)

    @staticmethod
    def _seed_ast_nodes(ctx: TestContext) -> None:
        """Seed AST nodes that create validation gaps.

        Creates an AST node without a corresponding GOID to trigger
        'ast_node_missing_goid' validation warnings.
        """
        ctx.gateway.con.execute(
            """
            INSERT INTO core.ast_nodes (
                path, node_type, name, qualname, lineno, end_lineno,
                col_offset, end_col_offset, parent_qualname, decorators,
                docstring, hash
            ) VALUES (
                'pkg/a.py', 'FunctionDef', 'foo', 'pkg.a.foo', 1, 2, 0, 0,
                'pkg.a', '[]', NULL, 'h1'
            )
            """
        )

    @staticmethod
    def _seed_modules(ctx: TestContext) -> None:
        """Seed the core.modules table."""
        rows = [
            ModuleRow(
                module="pkg.a",
                path="pkg/a.py",
                repo=REPO,
                commit=COMMIT,
            ),
            ModuleRow(
                module="pkg.b",
                path="pkg/b.py",
                repo=REPO,
                commit=COMMIT,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_goids(ctx: TestContext, now: datetime) -> None:
        """Seed GOIDs that reference missing call graph nodes.

        Creates a GOID for a caller that references a callee without
        a corresponding call graph node, triggering validation warnings.
        """
        rows = [
            GoidRow(
                goid_h128=CALLER_GOID,
                urn="urn:pkg.b.caller",
                repo=REPO,
                commit=COMMIT,
                rel_path="pkg/b.py",
                kind="function",
                qualname="pkg.b.caller",
                start_line=1,
                end_line=5,
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)


VALIDATION_PACK = ValidationPack()

__all__ = ["VALIDATION_PACK", "ValidationPack"]
