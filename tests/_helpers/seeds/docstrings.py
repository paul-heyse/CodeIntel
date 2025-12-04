"""Docstring seed pack for documentation data.

This module provides the DocstringPack which seeds the core.docstrings table
with parsed docstring data for functions and classes.

The pack depends on CORE_PACK and uses its GOID definitions to create
realistic docstring entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.builders import DocstringRow, insert_rows
from tests._helpers.seeds.core import (
    CORE_PACK,
    MOD_A_FQN,
    MOD_A_PATH,
    MOD_B_FQN,
    MOD_B_PATH,
    MOD_UTIL_FQN,
    MOD_UTIL_PATH,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Docstring Pack Implementation
# =============================================================================


@dataclass
class DocstringPack:
    """Seed pack for docstring data.

    Seeds core.docstrings table with parsed docstring entries for functions
    defined in CORE_PACK.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_params : bool
        Whether to include parameter documentation.
    include_returns : bool
        Whether to include return documentation.
    include_raises : bool
        Whether to include raises documentation.
    """

    name: str = "docstrings"
    include_params: bool = True
    include_returns: bool = True
    include_raises: bool = True

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for module/GOID data.
        """
        return (CORE_PACK,)

    def apply(self, ctx: TestContext) -> None:
        """Apply docstring seeds to the test context.

        Seeds core.docstrings with parsed docstring entries.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)
        self._seed_docstrings(ctx, now)

    def _seed_docstrings(self, ctx: TestContext, now: datetime) -> None:
        """Seed the docstrings table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        params_json = "[]" if not self.include_params else '[{"name": "x", "type": "int"}]'
        returns_json = "null" if not self.include_returns else '{"type": "int", "desc": "Result"}'
        raises_json = "[]" if not self.include_raises else '[{"type": "ValueError"}]'

        rows = [
            DocstringRow(
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_A_PATH,
                module=MOD_A_FQN,
                qualname="func_a",
                kind="function",
                lineno=1,
                end_lineno=10,
                raw_docstring="Add two numbers and return the sum.",
                style="numpy",
                short_desc="Add two numbers and return the sum.",
                long_desc="This function demonstrates basic arithmetic operations.",
                params_json=params_json,
                returns_json=returns_json,
                raises_json=raises_json,
                examples_json="[]",
                created_at=now,
            ),
            DocstringRow(
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_B_PATH,
                module=MOD_B_FQN,
                qualname="func_b",
                kind="function",
                lineno=1,
                end_lineno=15,
                raw_docstring="Double a number and call func_c.",
                style="numpy",
                short_desc="Double a number and call func_c.",
                long_desc="",
                params_json=params_json,
                returns_json=returns_json,
                raises_json="[]",
                examples_json="[]",
                created_at=now,
            ),
            DocstringRow(
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_UTIL_PATH,
                module=MOD_UTIL_FQN,
                qualname="helper",
                kind="function",
                lineno=1,
                end_lineno=5,
                raw_docstring="Return the value unchanged.",
                style="numpy",
                short_desc="Return the value unchanged.",
                long_desc="A simple passthrough helper function.",
                params_json=params_json,
                returns_json=returns_json,
                raises_json="[]",
                examples_json='[{"code": "helper(5)", "output": "5"}]',
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)


# Default instance for common usage
DOCSTRING_PACK = DocstringPack()


__all__ = [
    "DOCSTRING_PACK",
    "DocstringPack",
]
