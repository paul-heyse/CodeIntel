"""Pure data-flow graph construction.

This module provides stateless functions for building data-flow graphs
from CFG blocks without any database or file I/O.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.data_models.rows import DFGEdgeRow

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.graphs.compute.cfg import BasicBlock, CFGEdge


@dataclass(frozen=True)
class DFGEdge:
    """Represents a data-flow edge.

    Attributes
    ----------
    function_goid
        GOID of the containing function.
    src_block_idx
        Source block index.
    dst_block_idx
        Destination block index.
    src_var
        Source variable name.
    dst_var
        Destination variable name.
    edge_kind
        Edge kind (data-flow, intra-block).
    via_phi
        Whether edge passes through a phi node.
    use_kind
        Use kind descriptor.
    """

    function_goid: int
    src_block_idx: int
    dst_block_idx: int
    src_var: str
    dst_var: str
    edge_kind: str
    via_phi: bool
    use_kind: str


@dataclass(frozen=True)
class DFGResult:
    """Result of DFG construction.

    Attributes
    ----------
    edges
        Data-flow edges.
    function_goid
        GOID of the function.
    """

    edges: tuple[DFGEdge, ...]
    function_goid: int


class DFGBuilder:
    """Builds DFG from CFG blocks and edges.

    Uses reaching definitions analysis to construct data-flow edges.
    """

    def __init__(
        self,
        goid: int,
        blocks: Sequence[BasicBlock],
        cfg_edges: Sequence[CFGEdge],
    ) -> None:
        """Initialize the DFG builder.

        Parameters
        ----------
        goid
            Function GOID.
        blocks
            CFG basic blocks.
        cfg_edges
            CFG edges.
        """
        self.goid = goid
        self.blocks = list(blocks)
        self.cfg_edges = list(cfg_edges)
        self.dfg_edges: list[DFGEdge] = []

        self.preds: dict[int, list[int]] = defaultdict(list)
        for e in cfg_edges:
            self.preds[e.dst].append(e.src)

    def build(self) -> DFGResult:
        """Construct DFG edges using reaching definitions.

        Returns
        -------
        DFGResult
            Data-flow edges.
        """
        block_defs = self._collect_block_defs()
        reach_in = self._compute_reaching_defs(block_defs)
        for block in self.blocks:
            self._emit_edges_for_block(block, reach_in[block.idx])
        return DFGResult(edges=tuple(self.dfg_edges), function_goid=self.goid)

    def _collect_block_defs(self) -> dict[int, dict[str, str]]:
        """Collect variable definitions per block.

        Returns
        -------
        dict[int, dict[str, str]]
            Block index to variable definitions mapping.
        """
        block_defs: dict[int, dict[str, str]] = {}
        for block in self.blocks:
            defs: dict[str, str] = {}
            for stmt in block.stmts:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Name):
                        if isinstance(node.ctx, ast.Store):
                            defs[node.id] = "assignment"
                    elif isinstance(node, ast.arg):
                        defs[node.arg] = "param"
            block_defs[block.idx] = defs
        return block_defs

    def _compute_reaching_defs(
        self,
        block_defs: dict[int, dict[str, str]],
    ) -> dict[int, dict[str, set[int]]]:
        """Compute reaching definitions for all blocks.

        Parameters
        ----------
        block_defs
            Variable definitions per block.

        Returns
        -------
        dict[int, dict[str, set[int]]]
            Block index to reaching definitions mapping.
        """
        reach_in: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        changed = True
        while changed:
            changed = False
            for block in self.blocks:
                new_in: dict[str, set[int]] = defaultdict(set)
                for pred_idx in self.preds[block.idx]:
                    pred_defs = block_defs.get(pred_idx, {})
                    pred_reach = reach_in[pred_idx]
                    all_syms = set(pred_defs) | set(pred_reach)
                    for sym in all_syms:
                        if sym in pred_defs:
                            new_in[sym].add(pred_idx)
                        else:
                            new_in[sym].update(pred_reach.get(sym, set()))
                if new_in != reach_in[block.idx]:
                    reach_in[block.idx] = new_in
                    changed = True
        return reach_in

    def _emit_edges_for_block(
        self,
        block: BasicBlock,
        reaching_defs: Mapping[str, set[int]],
    ) -> None:
        """Emit DFG edges for a block.

        Parameters
        ----------
        block
            Basic block to process.
        reaching_defs
            Reaching definitions at block entry.
        """
        current_defs = dict(reaching_defs)
        local_defs: set[str] = set()
        for stmt in block.stmts:
            stmt_uses, stmt_defs = self._collect_stmt_symbols(stmt)
            self._emit_use_edges(block.idx, stmt_uses, current_defs, local_defs)
            local_defs.update(stmt_defs)

    @staticmethod
    def _collect_stmt_symbols(stmt: ast.AST) -> tuple[list[str], list[str]]:
        """Collect uses and definitions from a statement.

        Parameters
        ----------
        stmt
            AST statement.

        Returns
        -------
        tuple[list[str], list[str]]
            (uses, definitions) lists.
        """
        uses: list[str] = []
        defs: list[str] = []
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    uses.append(node.id)
                elif isinstance(node.ctx, ast.Store):
                    defs.append(node.id)
            elif isinstance(node, ast.arg):
                defs.append(node.arg)
        return uses, defs

    def _emit_use_edges(
        self,
        block_idx: int,
        uses: Sequence[str],
        reaching_defs: Mapping[str, set[int]],
        local_defs: set[str],
    ) -> None:
        """Emit edges for variable uses.

        Parameters
        ----------
        block_idx
            Current block index.
        uses
            Variable uses.
        reaching_defs
            Reaching definitions.
        local_defs
            Variables defined locally in this block.
        """
        for sym in uses:
            if sym in local_defs:
                self.dfg_edges.append(
                    DFGEdge(
                        function_goid=self.goid,
                        src_block_idx=block_idx,
                        dst_block_idx=block_idx,
                        src_var=sym,
                        dst_var=sym,
                        edge_kind="intra-block",
                        via_phi=False,
                        use_kind="intra-block",
                    )
                )
                continue
            for src_idx in reaching_defs.get(sym, set()):
                self.dfg_edges.append(
                    DFGEdge(
                        function_goid=self.goid,
                        src_block_idx=src_idx,
                        dst_block_idx=block_idx,
                        src_var=sym,
                        dst_var=sym,
                        edge_kind="data-flow",
                        via_phi=False,
                        use_kind="data-flow",
                    )
                )


def build_dfg(
    goid: int,
    blocks: Sequence[BasicBlock],
    cfg_edges: Sequence[CFGEdge],
) -> DFGResult:
    """Build a data-flow graph from CFG.

    Parameters
    ----------
    goid
        Function GOID.
    blocks
        CFG basic blocks.
    cfg_edges
        CFG edges.

    Returns
    -------
    DFGResult
        Constructed DFG.
    """
    builder = DFGBuilder(goid, blocks, cfg_edges)
    return builder.build()


def dfg_to_rows(
    result: DFGResult,
    repo: str,
    commit: str,
) -> Sequence[DFGEdgeRow]:
    """Convert DFG result to database rows.

    Parameters
    ----------
    result
        DFG construction result.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    Sequence[DFGEdgeRow]
        Edge rows for persistence.
    """
    return [
        DFGEdgeRow(
            repo=repo,
            commit=commit,
            function_goid_h128=result.function_goid,
            src_block_id=f"{result.function_goid}:block{edge.src_block_idx}",
            dst_block_id=f"{result.function_goid}:block{edge.dst_block_idx}",
            src_var=edge.src_var,
            dst_var=edge.dst_var,
            edge_kind=edge.edge_kind,
            via_phi=edge.via_phi,
            use_kind=edge.use_kind,
        )
        for edge in result.edges
    ]


__all__ = [
    "DFGBuilder",
    "DFGEdge",
    "DFGEdgeRow",
    "DFGResult",
    "build_dfg",
    "dfg_to_rows",
]
