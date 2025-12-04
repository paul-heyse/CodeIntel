"""Graph seed pack for call graph and import graph data.

This module provides the GraphPack which seeds graph-related tables:
call graph nodes, call graph edges, import graph edges, CFG blocks/edges,
and DFG edges.

The pack depends on CORE_PACK and uses its module and GOID definitions
to create realistic graph relationships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    ImportGraphEdgeRow,
    insert_rows,
)
from tests._helpers.seeds.core import (
    CORE_PACK,
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_FQN,
    MOD_A_PATH,
    MOD_B_FQN,
    MOD_B_PATH,
    MOD_C_FQN,
    MOD_UTIL_FQN,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Graph Data Constants
# =============================================================================

# CFG block identifiers
CFG_ENTRY = "entry"
CFG_BODY = "body"
CFG_EXIT = "exit"


# =============================================================================
# Graph Pack Implementation
# =============================================================================


@dataclass
class GraphPack:
    """Seed pack for graph structure data.

    Seeds call graph, import graph, and CFG/DFG tables with consistent
    test data. Creates realistic graph relationships between GOIDs
    and modules defined by CORE_PACK.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_call_graph : bool
        Whether to seed call graph data.
    include_import_graph : bool
        Whether to seed import graph data.
    include_cfg : bool
        Whether to seed CFG blocks and edges.
    include_dfg : bool
        Whether to seed DFG edges.
    """

    name: str = "graph"
    include_call_graph: bool = True
    include_import_graph: bool = True
    include_cfg: bool = True
    include_dfg: bool = True

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
        """Apply graph seeds to the test context.

        Seeds call graph, import graph, and CFG/DFG tables.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        if self.include_call_graph:
            self._seed_call_graph(ctx)

        if self.include_import_graph:
            self._seed_import_graph(ctx)

        if self.include_cfg:
            self._seed_cfg(ctx)

        if self.include_dfg:
            self._seed_dfg(ctx)

    @staticmethod
    def _seed_call_graph(ctx: TestContext) -> None:
        """Seed call graph nodes and edges.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        # Call graph nodes for each function
        nodes = [
            CallGraphNodeRow(
                goid_h128=GOID_FUNC_A,
                language="python",
                kind="function",
                arity=2,
                is_public=True,
                rel_path=MOD_A_PATH,
            ),
            CallGraphNodeRow(
                goid_h128=GOID_FUNC_B,
                language="python",
                kind="function",
                arity=1,
                is_public=True,
                rel_path=MOD_B_PATH,
            ),
            CallGraphNodeRow(
                goid_h128=GOID_FUNC_C,
                language="python",
                kind="function",
                arity=0,
                is_public=False,
                rel_path=MOD_A_PATH,
            ),
            CallGraphNodeRow(
                goid_h128=GOID_HELPER,
                language="python",
                kind="function",
                arity=1,
                is_public=True,
                rel_path=MOD_A_PATH,
            ),
        ]
        insert_rows(ctx.gateway, nodes)

        # Call graph edges: func_a -> func_b, func_a -> helper, func_b -> func_c
        edges = [
            CallGraphEdgeRow(
                repo=ctx.repo,
                commit=ctx.commit,
                caller_goid_h128=GOID_FUNC_A,
                callee_goid_h128=GOID_FUNC_B,
                callsite_path=MOD_A_PATH,
                callsite_line=5,
                callsite_col=4,
                language="python",
                kind="call",
                resolved_via="static",
                confidence=1.0,
            ),
            CallGraphEdgeRow(
                repo=ctx.repo,
                commit=ctx.commit,
                caller_goid_h128=GOID_FUNC_A,
                callee_goid_h128=GOID_HELPER,
                callsite_path=MOD_A_PATH,
                callsite_line=7,
                callsite_col=4,
                language="python",
                kind="call",
                resolved_via="static",
                confidence=1.0,
            ),
            CallGraphEdgeRow(
                repo=ctx.repo,
                commit=ctx.commit,
                caller_goid_h128=GOID_FUNC_B,
                callee_goid_h128=GOID_FUNC_C,
                callsite_path=MOD_B_PATH,
                callsite_line=10,
                callsite_col=8,
                language="python",
                kind="call",
                resolved_via="static",
                confidence=0.9,
            ),
        ]
        insert_rows(ctx.gateway, edges)

    @staticmethod
    def _seed_import_graph(ctx: TestContext) -> None:
        """Seed import graph edges.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        # Import relationships: mod_a imports mod_b, mod_b imports mod_c and util
        edges = [
            ImportGraphEdgeRow(
                repo=ctx.repo,
                commit=ctx.commit,
                src_module=MOD_A_FQN,
                dst_module=MOD_B_FQN,
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=0,
            ),
            ImportGraphEdgeRow(
                repo=ctx.repo,
                commit=ctx.commit,
                src_module=MOD_B_FQN,
                dst_module=MOD_C_FQN,
                src_fan_out=2,
                dst_fan_in=1,
                cycle_group=0,
            ),
            ImportGraphEdgeRow(
                repo=ctx.repo,
                commit=ctx.commit,
                src_module=MOD_B_FQN,
                dst_module=MOD_UTIL_FQN,
                src_fan_out=2,
                dst_fan_in=1,
                cycle_group=0,
            ),
        ]
        insert_rows(ctx.gateway, edges)

    @staticmethod
    def _seed_cfg(ctx: TestContext) -> None:
        """Seed CFG blocks and edges.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        # CFG blocks for func_a: entry -> body -> exit
        blocks = [
            CFGBlockRow(
                function_goid_h128=GOID_FUNC_A,
                block_idx=0,
                block_id=f"{GOID_FUNC_A}:{CFG_ENTRY}",
                label=CFG_ENTRY,
                file_path=MOD_A_PATH,
                start_line=1,
                end_line=2,
                kind="entry",
                stmts_json="[]",
                in_degree=0,
                out_degree=1,
            ),
            CFGBlockRow(
                function_goid_h128=GOID_FUNC_A,
                block_idx=1,
                block_id=f"{GOID_FUNC_A}:{CFG_BODY}",
                label=CFG_BODY,
                file_path=MOD_A_PATH,
                start_line=3,
                end_line=8,
                kind="basic",
                stmts_json='["call", "return"]',
                in_degree=1,
                out_degree=1,
            ),
            CFGBlockRow(
                function_goid_h128=GOID_FUNC_A,
                block_idx=2,
                block_id=f"{GOID_FUNC_A}:{CFG_EXIT}",
                label=CFG_EXIT,
                file_path=MOD_A_PATH,
                start_line=9,
                end_line=10,
                kind="exit",
                stmts_json="[]",
                in_degree=1,
                out_degree=0,
            ),
        ]
        insert_rows(ctx.gateway, blocks)

        # CFG edges: entry -> body -> exit
        edges = [
            CFGEdgeRow(
                function_goid_h128=GOID_FUNC_A,
                src_block_id=f"{GOID_FUNC_A}:{CFG_ENTRY}",
                dst_block_id=f"{GOID_FUNC_A}:{CFG_BODY}",
                edge_kind="fallthrough",
            ),
            CFGEdgeRow(
                function_goid_h128=GOID_FUNC_A,
                src_block_id=f"{GOID_FUNC_A}:{CFG_BODY}",
                dst_block_id=f"{GOID_FUNC_A}:{CFG_EXIT}",
                edge_kind="fallthrough",
            ),
        ]
        insert_rows(ctx.gateway, edges)

    @staticmethod
    def _seed_dfg(ctx: TestContext) -> None:
        """Seed DFG edges.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        # DFG edges: data flow from entry to body to exit
        edges = [
            DFGEdgeRow(
                function_goid_h128=GOID_FUNC_A,
                src_block_id=f"{GOID_FUNC_A}:{CFG_ENTRY}",
                dst_block_id=f"{GOID_FUNC_A}:{CFG_BODY}",
                src_var="x",
                dst_var="x",
                edge_kind="def-use",
                via_phi=False,
                use_kind="read",
            ),
            DFGEdgeRow(
                function_goid_h128=GOID_FUNC_A,
                src_block_id=f"{GOID_FUNC_A}:{CFG_BODY}",
                dst_block_id=f"{GOID_FUNC_A}:{CFG_EXIT}",
                src_var="result",
                dst_var="result",
                edge_kind="def-use",
                via_phi=False,
                use_kind="return",
            ),
        ]
        insert_rows(ctx.gateway, edges)


# Default instance for common usage
GRAPH_PACK = GraphPack()


__all__ = [
    "CFG_BODY",
    "CFG_ENTRY",
    "CFG_EXIT",
    "GRAPH_PACK",
    "GraphPack",
]
