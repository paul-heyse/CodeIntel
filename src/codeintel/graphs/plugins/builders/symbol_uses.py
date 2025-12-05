"""Symbol uses builder plugin.

This module builds symbol usage graph from SCIP data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import SymbolUsesStepConfig
from codeintel.graphs.compute.symbol_uses import build_symbol_uses_data

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def build_scip_candidates(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, tuple[str, ...]]:
    """Build SCIP candidate lookup for call graph resolution.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping of symbol use to candidate qualnames.
    """
    con = gateway.con
    rows = con.execute(
        """
        SELECT use_symbol, array_agg(def_qualname)
        FROM graphs.symbol_uses
        WHERE repo = ? AND commit = ?
        GROUP BY use_symbol
        """,
        [repo, commit],
    ).fetchall()
    return {row[0]: tuple(row[1]) for row in rows}


class SymbolUsesPlugin(TargetPlugin):
    """Build symbol usage graph from SCIP data.

    Outputs
    -------
    - graphs.symbol_uses: Symbol usage relationships
    """

    plugin_name: ClassVar[str] = "symbol_uses"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build symbol usage graph from SCIP data."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute symbol uses construction.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result.
        """
        _ = self  # Protocol method requires instance

        cfg = SymbolUsesStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        try:
            row_counts = build_symbol_uses_data(ctx.gateway, cfg)
            return TargetResult.succeeded(row_counts=row_counts)
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Symbol uses build failed: {e}")


__all__ = ["SymbolUsesPlugin", "build_scip_candidates"]
