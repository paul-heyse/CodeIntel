"""Pure control-flow graph construction.

This module provides stateless functions for building control-flow graphs
from parsed AST nodes without any database or file I/O.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.core.data_models.rows import CFGBlockRow, CFGEdgeRow
from codeintel.core.serialization.payload import encode_payload

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class BasicBlock:
    """Represents a basic block in a CFG.

    Attributes
    ----------
    idx
        Block index.
    kind
        Block kind (entry, body, exit, loop_head, loop_exit).
    label
        Human-readable label.
    stmts
        AST statements in this block.
    start_line
        Starting line number.
    end_line
        Ending line number.
    """

    idx: int
    kind: str = "body"
    label: str = ""
    stmts: list[ast.AST] = field(default_factory=list)
    start_line: int = -1
    end_line: int = -1

    def stmt_kinds(self) -> list[str]:
        """Return statement type names for this block.

        Returns
        -------
        list[str]
            Statement type names in evaluation order.
        """
        return [type(stmt).__name__ for stmt in self.stmts]


@dataclass(frozen=True)
class CFGEdge:
    """Represents an edge in a CFG.

    Attributes
    ----------
    src
        Source block index.
    dst
        Destination block index.
    kind
        Edge kind (fallthrough, true, false, loop, back, jump).
    """

    src: int
    dst: int
    kind: str


@dataclass(frozen=True)
class CFGResult:
    """Result of CFG construction.

    Attributes
    ----------
    blocks
        Basic blocks in the CFG.
    edges
        Edges connecting blocks.
    function_goid
        GOID of the function.
    """

    blocks: tuple[BasicBlock, ...]
    edges: tuple[CFGEdge, ...]
    function_goid: int


class CFGBuilder:
    """Builds CFG for a single function AST.

    This class constructs a control-flow graph from a function's AST,
    identifying basic blocks and control-flow edges.
    """

    def __init__(
        self,
        goid: int,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
    ) -> None:
        """Initialize the CFG builder.

        Parameters
        ----------
        goid
            Function GOID.
        func_node
            Function AST node.
        file_path
            Source file path.
        """
        self.goid = goid
        self.func_node = func_node
        self.file_path = file_path
        self.blocks: list[BasicBlock] = []
        self.edges: list[CFGEdge] = []
        self.current_block: BasicBlock | None = None
        self.loop_stack: list[tuple[int, int]] = []

    def new_block(self, kind: str = "body", label: str = "") -> BasicBlock:
        """Create and register a new basic block.

        Parameters
        ----------
        kind
            Block kind.
        label
            Block label.

        Returns
        -------
        BasicBlock
            The newly created block.
        """
        idx = len(self.blocks)
        if not label:
            label = f"{kind}:{idx}"
        block = BasicBlock(idx, kind, label)
        self.blocks.append(block)
        return block

    def add_edge(self, src: int, dst: int, kind: str = "fallthrough") -> None:
        """Add a directed edge between blocks.

        Parameters
        ----------
        src
            Source block index.
        dst
            Destination block index.
        kind
            Edge kind.
        """
        self.edges.append(CFGEdge(src, dst, kind))

    def build(self) -> CFGResult:
        """Construct the CFG.

        Returns
        -------
        CFGResult
            Blocks and edges representing the CFG.
        """
        entry = self.new_block("entry")
        entry.start_line = self.func_node.lineno
        entry.end_line = self.func_node.lineno

        self.current_block = self.new_block("body")
        self.add_edge(entry.idx, self.current_block.idx)

        for stmt in self.func_node.body:
            self.visit(stmt)

        exit_block = self.new_block("exit")
        end_lineno = getattr(self.func_node, "end_lineno", -1)
        exit_block.start_line = end_lineno if end_lineno is not None else -1
        exit_block.end_line = exit_block.start_line

        if self.current_block:
            self.add_edge(self.current_block.idx, exit_block.idx)

        return CFGResult(
            blocks=tuple(self.blocks),
            edges=tuple(self.edges),
            function_goid=self.goid,
        )

    def visit(self, node: ast.AST) -> None:
        """Dispatch visit to specific node handlers.

        Parameters
        ----------
        node
            AST node to visit.
        """
        if self.current_block is None:
            self.current_block = self.new_block()

        if hasattr(node, "lineno"):
            lineno = getattr(node, "lineno", -1)
            if self.current_block.start_line == -1:
                self.current_block.start_line = lineno
            self.current_block.end_line = getattr(node, "end_lineno", lineno)

        if isinstance(node, ast.If):
            self._visit_if(node)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            self._visit_loop(node)
        elif isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            self.current_block.stmts.append(node)
            self._visit_jump(node)
        elif isinstance(node, ast.Try):
            self._visit_try(node)
        elif isinstance(node, ast.ExceptHandler):
            for stmt in node.body:
                self.visit(stmt)
        else:
            self.current_block.stmts.append(node)

    def _visit_try(self, node: ast.Try) -> None:
        """Handle try statement.

        Parameters
        ----------
        node
            Try AST node.
        """
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            self.visit(handler)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)

    def _visit_if(self, node: ast.If) -> None:
        """Handle if statement.

        Parameters
        ----------
        node
            If AST node.
        """
        cond_block = self.current_block
        if cond_block is None:
            return
        cond_block.stmts.append(node.test)
        self.current_block = None

        true_entry = self.new_block("body", "if_true")
        self.add_edge(cond_block.idx, true_entry.idx, "true")
        self.current_block = true_entry
        for stmt in node.body:
            self.visit(stmt)
        true_exit = self.current_block

        false_exit = None
        if node.orelse:
            false_entry = self.new_block("body", "if_false")
            self.add_edge(cond_block.idx, false_entry.idx, "false")
            self.current_block = false_entry
            for stmt in node.orelse:
                self.visit(stmt)
            false_exit = self.current_block

        join_block = self.new_block("body", "if_join")

        if true_exit:
            self.add_edge(true_exit.idx, join_block.idx)

        if false_exit:
            self.add_edge(false_exit.idx, join_block.idx)
        elif not node.orelse:
            self.add_edge(cond_block.idx, join_block.idx, "false")

        self.current_block = join_block

    def _visit_loop(self, node: ast.For | ast.AsyncFor | ast.While) -> None:
        """Handle loop statement.

        Parameters
        ----------
        node
            Loop AST node.
        """
        pre_loop = self.current_block
        self.current_block = None

        loop_head = self.new_block("loop_head")
        if pre_loop:
            self.add_edge(pre_loop.idx, loop_head.idx)

        if isinstance(node, (ast.For, ast.AsyncFor)):
            loop_head.stmts.append(node.target)
            loop_head.stmts.append(node.iter)
        elif isinstance(node, ast.While):
            loop_head.stmts.append(node.test)

        loop_exit = self.new_block("loop_exit")
        self.loop_stack.append((loop_head.idx, loop_exit.idx))

        body_entry = self.new_block("body", "loop_body")
        self.add_edge(loop_head.idx, body_entry.idx, "loop")
        self.current_block = body_entry
        for stmt in node.body:
            self.visit(stmt)

        if self.current_block:
            self.add_edge(self.current_block.idx, loop_head.idx, "back")

        if node.orelse:
            orelse_entry = self.new_block("body", "loop_else")
            self.add_edge(loop_head.idx, orelse_entry.idx, "false")
            self.current_block = orelse_entry
            for stmt in node.orelse:
                self.visit(stmt)
            if self.current_block:
                self.add_edge(self.current_block.idx, loop_exit.idx)
        else:
            self.add_edge(loop_head.idx, loop_exit.idx, "false")

        self.loop_stack.pop()
        self.current_block = loop_exit

    def _visit_jump(
        self,
        node: ast.Return | ast.Raise | ast.Break | ast.Continue,
    ) -> None:
        """Handle jump statement.

        Parameters
        ----------
        node
            Jump AST node.
        """
        if self.current_block is None:
            return

        if isinstance(node, (ast.Break, ast.Continue)):
            if not self.loop_stack:
                return
            head_idx, exit_idx = self.loop_stack[-1]
            target = head_idx if isinstance(node, ast.Continue) else exit_idx
            self.add_edge(self.current_block.idx, target, "jump")

        self.current_block = None


def build_cfg(
    goid: int,
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
) -> CFGResult:
    """Build a control-flow graph for a function.

    Parameters
    ----------
    goid
        Function GOID.
    func_node
        Function AST node.
    file_path
        Source file path.

    Returns
    -------
    CFGResult
        Constructed CFG.
    """
    builder = CFGBuilder(goid, func_node, file_path)
    return builder.build()


def cfg_to_rows(
    result: CFGResult,
    file_path: str,
    default_start: int,
    default_end: int,
) -> tuple[Sequence[CFGBlockRow], Sequence[CFGEdgeRow]]:
    """Convert CFG result to database rows.

    Parameters
    ----------
    result
        CFG construction result.
    file_path
        Source file path.
    default_start
        Default start line for blocks with -1.
    default_end
        Default end line for blocks with -1.

    Returns
    -------
    tuple[Sequence[CFGBlockRow], Sequence[CFGEdgeRow]]
        Block and edge rows for persistence.
    """
    in_degree: dict[int, int] = {}
    out_degree: dict[int, int] = {}
    for edge in result.edges:
        out_degree[edge.src] = out_degree.get(edge.src, 0) + 1
        in_degree[edge.dst] = in_degree.get(edge.dst, 0) + 1

    block_rows = [
        CFGBlockRow(
            function_goid_h128=result.function_goid,
            block_idx=block.idx,
            block_id=f"{result.function_goid}:block{block.idx}",
            label=block.label,
            file_path=file_path,
            start_line=default_start if block.start_line == -1 else block.start_line,
            end_line=default_end if block.end_line == -1 else block.end_line,
            kind=block.kind,
            stmts_json=encode_payload(block.stmt_kinds()) or b"",
            in_degree=in_degree.get(block.idx, 0),
            out_degree=out_degree.get(block.idx, 0),
        )
        for block in result.blocks
    ]

    edge_rows = [
        CFGEdgeRow(
            function_goid_h128=result.function_goid,
            src_block_id=f"{result.function_goid}:block{edge.src}",
            dst_block_id=f"{result.function_goid}:block{edge.dst}",
            edge_kind=edge.kind,
        )
        for edge in result.edges
    ]

    return block_rows, edge_rows


__all__ = [
    "BasicBlock",
    "CFGBlockRow",
    "CFGBuilder",
    "CFGEdge",
    "CFGEdgeRow",
    "CFGResult",
    "build_cfg",
    "cfg_to_rows",
]
