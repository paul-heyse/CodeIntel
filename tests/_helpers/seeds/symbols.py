"""Symbol seed pack for symbol use edge data.

This module provides the SymbolPack which seeds the graph.symbol_use_edges
table with symbol definition and usage relationships.

The pack depends on CORE_PACK and uses its module and GOID definitions to
create realistic symbol reference patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.builders import (
    SymbolEdgeOptions,
    insert_symbol_use_edges,
    make_symbol_use_edge_row,
)
from tests._helpers.seeds.core import (
    CORE_PACK,
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_PATH,
    MOD_B_PATH,
    MOD_C_PATH,
    MOD_UTIL_PATH,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Symbol Pack Implementation
# =============================================================================


@dataclass
class SymbolPack:
    """Seed pack for symbol use edge data.

    Seeds graph.symbol_use_edges table with symbol definition and usage
    relationships based on modules and GOIDs from CORE_PACK.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_goids : bool
        Whether to include GOID references in symbol edges.
    """

    name: str = "symbols"
    include_goids: bool = True

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
        """Apply symbol seeds to the test context.

        Seeds graph.symbol_use_edges with symbol usage relationships.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        self._seed_symbol_use_edges(ctx)

    def _seed_symbol_use_edges(self, ctx: TestContext) -> None:
        """Seed the symbol_use_edges table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        # Symbol usage patterns:
        # - func_a is defined in mod_a, used in mod_b
        # - func_b is defined in mod_b, used in mod_a and mod_c
        # - func_c is defined in mod_c, used in mod_b
        # - helper is defined in util, used in mod_a

        rows = [
            # func_b used in mod_a (cross-module)
            make_symbol_use_edge_row(
                "func_b",
                MOD_B_PATH,
                MOD_A_PATH,
                options=SymbolEdgeOptions(
                    same_file=False,
                    same_module=False,
                    def_goid_h128=GOID_FUNC_B if self.include_goids else None,
                    use_goid_h128=GOID_FUNC_A if self.include_goids else None,
                ),
            ),
            # func_c used in mod_b (cross-module)
            make_symbol_use_edge_row(
                "func_c",
                MOD_C_PATH,
                MOD_B_PATH,
                options=SymbolEdgeOptions(
                    same_file=False,
                    same_module=False,
                    def_goid_h128=GOID_FUNC_C if self.include_goids else None,
                    use_goid_h128=GOID_FUNC_B if self.include_goids else None,
                ),
            ),
            # helper used in mod_a (cross-module)
            make_symbol_use_edge_row(
                "helper",
                MOD_UTIL_PATH,
                MOD_A_PATH,
                options=SymbolEdgeOptions(
                    same_file=False,
                    same_module=False,
                    def_goid_h128=GOID_HELPER if self.include_goids else None,
                    use_goid_h128=GOID_FUNC_A if self.include_goids else None,
                ),
            ),
            # Internal symbol reference within mod_a
            make_symbol_use_edge_row(
                "_internal_a",
                MOD_A_PATH,
                MOD_A_PATH,
                options=SymbolEdgeOptions(
                    same_file=True,
                    same_module=True,
                    use_goid_h128=GOID_FUNC_A if self.include_goids else None,
                ),
            ),
            # Cross-file but same package reference
            make_symbol_use_edge_row(
                "MOD_CONSTANT",
                MOD_UTIL_PATH,
                MOD_B_PATH,
                options=SymbolEdgeOptions(
                    same_file=False,
                    same_module=False,
                    use_goid_h128=GOID_FUNC_B if self.include_goids else None,
                ),
            ),
        ]
        insert_symbol_use_edges(ctx.gateway, rows)


# Default instance for common usage
SYMBOL_PACK = SymbolPack()


__all__ = [
    "SYMBOL_PACK",
    "SymbolPack",
]
