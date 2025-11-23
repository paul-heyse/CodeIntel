"""Build control-flow and data-flow graphs for functions."""

from __future__ import annotations

import ast
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from codeintel.config.models import CFGBuilderConfig
from codeintel.graphs.function_catalog import load_function_catalog
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import (
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    cfg_block_to_tuple,
    cfg_edge_to_tuple,
    dfg_edge_to_tuple,
)

log = logging.getLogger(__name__)


@dataclass
class Block:
    """Internal representation of a basic block."""

    idx: int
    kind: str = "body"  # entry, body, exit
    label: str = ""
    stmts: list[ast.AST] = field(default_factory=list)
    start_line: int = -1
    end_line: int = -1

    def to_json(self) -> str:
        """
        Serialize statements to JSON for debugging/display.

        Returns
        -------
        str
            JSON list of statement type names.
        """
        return json.dumps([type(s).__name__ for s in self.stmts])


@dataclass
class Edge:
    """Internal representation of a CFG edge."""

    src: int
    dst: int
    kind: str  # fallthrough, true, false, loop, exception


class CFGBuilder:
    """Builds CFG for a single function AST."""

    def __init__(
        self, goid: int, func_node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: str
    ) -> None:
        self.goid = goid
        self.func_node = func_node
        self.file_path = file_path
        self.blocks: list[Block] = []
        self.edges: list[Edge] = []
        self.current_block: Block | None = None
        self.loop_stack: list[tuple[int, int]] = []  # (head_idx, exit_idx)

    def new_block(self, kind: str = "body", label: str = "") -> Block:
        """
        Create and register a new basic block.

        Returns
        -------
        Block
            The newly created block.
        """
        idx = len(self.blocks)
        if not label:
            label = f"{kind}:{idx}"
        block = Block(idx, kind, label)
        self.blocks.append(block)
        return block

    def add_edge(self, src: int, dst: int, kind: str = "fallthrough") -> None:
        """Add a directed edge between blocks."""
        self.edges.append(Edge(src, dst, kind))

    def build(self) -> tuple[list[Block], list[Edge]]:
        """
        Construct the CFG.

        Returns
        -------
        tuple[list[Block], list[Edge]]
            List of blocks and edges representing the CFG.
        """
        entry = self.new_block("entry")
        entry.start_line = self.func_node.lineno
        entry.end_line = self.func_node.lineno

        self.current_block = self.new_block("body")
        self.add_edge(entry.idx, self.current_block.idx)

        # Visit body
        for stmt in self.func_node.body:
            self.visit(stmt)

        # Ensure exit block
        exit_block = self.new_block("exit")
        end_lineno = getattr(self.func_node, "end_lineno", -1)
        exit_block.start_line = end_lineno if end_lineno is not None else -1
        exit_block.end_line = exit_block.start_line

        # Connect last block to exit if not already terminated (e.g. by return)
        if self.current_block:
            self.add_edge(self.current_block.idx, exit_block.idx)

        return self.blocks, self.edges

    def visit(self, node: ast.AST) -> None:
        """Dispatch visit to specific node handlers."""
        if self.current_block is None:
            self.current_block = self.new_block()

        # Update block span
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
        elif isinstance(node, (ast.Try,)):
            self._visit_try(node)
        elif isinstance(node, ast.ExceptHandler):
            for stmt in node.body:
                self.visit(stmt)
        else:
            # Sequential statement
            self.current_block.stmts.append(node)

    def _visit_try(self, node: ast.Try) -> None:
        # For now, just treat body as sequential, ignoring handlers flow complexity
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            self.visit(handler)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)

    def _visit_if(self, node: ast.If) -> None:
        # End current block (condition check)
        cond_block = self.current_block
        if cond_block is None:  # Should not happen due to visit() guard
            return
        cond_block.stmts.append(node.test)
        self.current_block = None

        # True branch
        true_entry = self.new_block("body", "if_true")
        self.add_edge(cond_block.idx, true_entry.idx, "true")
        self.current_block = true_entry
        for stmt in node.body:
            self.visit(stmt)
        true_exit = self.current_block

        # False branch
        false_exit = None
        if node.orelse:
            false_entry = self.new_block("body", "if_false")
            self.add_edge(cond_block.idx, false_entry.idx, "false")
            self.current_block = false_entry
            for stmt in node.orelse:
                self.visit(stmt)
            false_exit = self.current_block

        # Join
        join_block = self.new_block("body", "if_join")

        if true_exit:
            self.add_edge(true_exit.idx, join_block.idx)

        if false_exit:
            self.add_edge(false_exit.idx, join_block.idx)
        elif not node.orelse:
            # Direct edge from cond to join if false and no else
            self.add_edge(cond_block.idx, join_block.idx, "false")

        self.current_block = join_block

    def _visit_loop(self, node: ast.For | ast.AsyncFor | ast.While) -> None:
        # End current block -> loop head
        pre_loop = self.current_block
        self.current_block = None

        loop_head = self.new_block("loop_head")
        if pre_loop:
            self.add_edge(pre_loop.idx, loop_head.idx)

        # Add loop var/test to head
        if isinstance(node, (ast.For, ast.AsyncFor)):
            loop_head.stmts.append(node.target)
            loop_head.stmts.append(node.iter)
        elif isinstance(node, ast.While):
            loop_head.stmts.append(node.test)

        loop_exit = self.new_block("loop_exit")
        self.loop_stack.append((loop_head.idx, loop_exit.idx))

        # Body
        body_entry = self.new_block("body", "loop_body")
        self.add_edge(loop_head.idx, body_entry.idx, "loop")
        self.current_block = body_entry
        for stmt in node.body:
            self.visit(stmt)

        # Back edge
        if self.current_block:
            self.add_edge(self.current_block.idx, loop_head.idx, "back")

        # Orelse (loops can have else!)
        if node.orelse:
            orelse_entry = self.new_block("body", "loop_else")
            self.add_edge(loop_head.idx, orelse_entry.idx, "false")
            self.current_block = orelse_entry
            for stmt in node.orelse:
                self.visit(stmt)
            if self.current_block:
                self.add_edge(self.current_block.idx, loop_exit.idx)
        else:
            # Exit loop directly
            self.add_edge(loop_head.idx, loop_exit.idx, "false")

        self.loop_stack.pop()
        self.current_block = loop_exit

    def _visit_jump(self, node: ast.Return | ast.Raise | ast.Break | ast.Continue) -> None:
        if self.current_block is None:
            return

        if isinstance(node, (ast.Break, ast.Continue)):
            if not self.loop_stack:
                return  # Syntax error or outside loop context we track
            head_idx, exit_idx = self.loop_stack[-1]
            target = head_idx if isinstance(node, ast.Continue) else exit_idx
            self.add_edge(self.current_block.idx, target, "jump")
        else:
            # Return/Raise -> assume exits function
            pass
        self.current_block = None


class DFGBuilder:
    """Builds DFG from CFG blocks and AST."""

    def __init__(self, goid: int, blocks: list[Block], edges: list[Edge]) -> None:
        self.goid = goid
        self.blocks = blocks
        self.edges = edges
        self.dfg_edges: list[DFGEdgeRow] = []

        # Precompute preds
        self.preds: dict[int, list[int]] = defaultdict(list)
        for e in edges:
            self.preds[e.dst].append(e.src)

    def build(self) -> list[DFGEdgeRow]:
        """
        Construct DFG edges using reaching definitions.

        Returns
        -------
        list[DFGEdgeRow]
            Data flow edges linking definitions to uses.
        """
        block_defs = self._collect_block_defs()
        reach_in = self._compute_reaching_defs(block_defs)
        for block in self.blocks:
            self._emit_edges_for_block(block, reach_in[block.idx])
        return self.dfg_edges

    def _collect_block_defs(self) -> dict[int, dict[str, str]]:
        block_defs: dict[int, dict[str, str]] = {}
        for block in self.blocks:
            defs: dict[str, str] = {}
            for stmt in block.stmts:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Name):
                        if isinstance(node.ctx, ast.Store):
                            defs[node.id] = "assignment"
                        elif isinstance(node.ctx, ast.Load):
                            continue
                    elif isinstance(node, ast.arg):
                        defs[node.arg] = "param"
            block_defs[block.idx] = defs
        return block_defs

    def _compute_reaching_defs(
        self, block_defs: dict[int, dict[str, str]]
    ) -> dict[int, dict[str, set[int]]]:
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

    def _emit_edges_for_block(self, block: Block, reaching_defs: dict[str, set[int]]) -> None:
        current_defs = reaching_defs.copy()
        local_defs: set[str] = set()
        for stmt in block.stmts:
            stmt_uses, stmt_defs = self._collect_stmt_symbols(stmt)
            self._emit_use_edges(block.idx, stmt_uses, current_defs, local_defs)
            local_defs.update(stmt_defs)

    @staticmethod
    def _collect_stmt_symbols(stmt: ast.AST) -> tuple[list[str], list[str]]:
        stmt_uses: list[str] = []
        stmt_defs: list[str] = []
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    stmt_uses.append(node.id)
                elif isinstance(node.ctx, ast.Store):
                    stmt_defs.append(node.id)
            elif isinstance(node, ast.arg):
                stmt_defs.append(node.arg)
        return stmt_uses, stmt_defs

    def _emit_use_edges(
        self,
        block_idx: int,
        stmt_uses: list[str],
        reaching_defs: dict[str, set[int]],
        local_defs: set[str],
    ) -> None:
        for sym in stmt_uses:
            if sym in local_defs:
                self.dfg_edges.append(
                    DFGEdgeRow(
                        function_goid_h128=self.goid,
                        src_block_id=f"{self.goid}:block{block_idx}",
                        dst_block_id=f"{self.goid}:block{block_idx}",
                        src_var=sym,
                        dst_var=sym,
                        edge_kind="intra-block",
                    )
                )
                continue
            for src_idx in reaching_defs.get(sym, set()):
                self.dfg_edges.append(
                    DFGEdgeRow(
                        function_goid_h128=self.goid,
                        src_block_id=f"{self.goid}:block{src_idx}",
                        dst_block_id=f"{self.goid}:block{block_idx}",
                        src_var=sym,
                        dst_var=sym,
                        edge_kind="data-flow",
                    )
                )


@dataclass(frozen=True)
class FunctionBuildSpec:
    """Specification of a single function to build CFG/DFG rows for."""

    goid: int
    repo_root: Path
    rel_path: str
    lines: tuple[int, int]
    qualname: str


def _load_source(spec: FunctionBuildSpec, file_cache: dict[str, str]) -> str:
    if spec.rel_path not in file_cache:
        try:
            file_cache[spec.rel_path] = (spec.repo_root / spec.rel_path).read_text(encoding="utf8")
        except Exception:  # noqa: BLE001
            file_cache[spec.rel_path] = ""
    return file_cache[spec.rel_path]


def _parse_function_ast(source: str) -> ast.Module | None:
    if not source:
        return None
    try:
        return ast.parse(source)
    except Exception:  # noqa: BLE001
        return None


def _select_target_node(
    tree: ast.Module, start_line: int, end_line: int, qualname: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno == start_line:
                return node
            if qualname.endswith(node.name) and start_line <= node.lineno <= end_line:
                return node
    return None


def _build_cfg_for_function(
    spec: FunctionBuildSpec, file_cache: dict[str, str]
) -> tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]:
    start_line, end_line = spec.lines
    source = _load_source(spec, file_cache)
    tree = _parse_function_ast(source)
    if tree is None:
        return [], [], []

    target_node = _select_target_node(tree, start_line, end_line, spec.qualname)
    if target_node is None:
        return [], [], []

    cfg_builder = CFGBuilder(spec.goid, target_node, spec.rel_path)
    blocks, edges = cfg_builder.build()

    in_degree: dict[int, int] = defaultdict(int)
    out_degree: dict[int, int] = defaultdict(int)
    for e in edges:
        out_degree[e.src] += 1
        in_degree[e.dst] += 1

    block_rows = [
        CFGBlockRow(
            function_goid_h128=spec.goid,
            block_idx=b.idx,
            block_id=f"{spec.goid}:block{b.idx}",
            label=b.label,
            file_path=spec.rel_path,
            start_line=start_line if b.start_line == -1 else b.start_line,
            end_line=end_line if b.end_line == -1 else b.end_line,
            kind=b.kind,
            stmts_json=b.to_json(),
            in_degree=in_degree[b.idx],
            out_degree=out_degree[b.idx],
        )
        for b in blocks
    ]

    edge_rows = [
        CFGEdgeRow(
            function_goid_h128=spec.goid,
            src_block_id=f"{spec.goid}:block{e.src}",
            dst_block_id=f"{spec.goid}:block{e.dst}",
            edge_kind=e.kind,
        )
        for e in edges
    ]

    dfg_rows = DFGBuilder(spec.goid, blocks, edges).build()
    return block_rows, edge_rows, dfg_rows


def _flush(
    con: duckdb.DuckDBPyConnection,
    blocks: list[CFGBlockRow],
    cfg_edges: list[CFGEdgeRow],
    dfg_edges: list[DFGEdgeRow],
) -> None:
    if blocks:
        run_batch(
            con,
            "graph.cfg_blocks",
            [cfg_block_to_tuple(r) for r in blocks],
            delete_params=[],  # Append only
            scope="cfg_blocks",
        )
    if cfg_edges:
        run_batch(
            con,
            "graph.cfg_edges",
            [cfg_edge_to_tuple(r) for r in cfg_edges],
            scope="cfg_edges",
        )
    if dfg_edges:
        run_batch(
            con,
            "graph.dfg_edges",
            [dfg_edge_to_tuple(r) for r in dfg_edges],
            scope="dfg_edges",
        )


def build_cfg_and_dfg(con: duckdb.DuckDBPyConnection, cfg: CFGBuilderConfig) -> None:
    """Emit CFG and DFG edges for each function GOID."""
    # Clear existing data for this repo/commit to allow idempotent re-runs
    log.info("Clearing existing CFG/DFG data for %s@%s", cfg.repo, cfg.commit)
    con.execute(
        """
        DELETE FROM graph.cfg_blocks
        WHERE function_goid_h128 IN (
            SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?
        )
        """,
        [cfg.repo, cfg.commit],
    )
    con.execute(
        """
        DELETE FROM graph.cfg_edges
        WHERE function_goid_h128 IN (
            SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?
        )
        """,
        [cfg.repo, cfg.commit],
    )
    con.execute(
        """
        DELETE FROM graph.dfg_edges
        WHERE function_goid_h128 IN (
            SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?
        )
        """,
        [cfg.repo, cfg.commit],
    )

    function_spans = load_function_catalog(con, repo=cfg.repo, commit=cfg.commit).function_spans
    if not function_spans:
        log.warning("No function GOIDs found; skipping CFG/DFG build.")
        return

    log.info("Building CFG/DFG for %d functions...", len(function_spans))

    all_blocks: list[CFGBlockRow] = []
    all_cfg_edges: list[CFGEdgeRow] = []
    all_dfg_edges: list[DFGEdgeRow] = []

    file_cache: dict[str, str] = {}

    for span in function_spans:
        start = span.start_line
        end = span.end_line

        spec = FunctionBuildSpec(
            goid=span.goid,
            repo_root=cfg.repo_root,
            rel_path=span.rel_path,
            lines=(start, end),
            qualname=span.qualname,
        )
        blocks, edges, dfg = _build_cfg_for_function(spec, file_cache)

        all_blocks.extend(blocks)
        all_cfg_edges.extend(edges)
        all_dfg_edges.extend(dfg)

    _flush(con, all_blocks, all_cfg_edges, all_dfg_edges)

    log.info("CFG/DFG build complete.")
