"""AST and CST-based call edge collection strategies.

This module consolidates call_ast.py and call_cst.py, providing
a unified interface for collecting call graph edges from Python source.
"""

from __future__ import annotations

import ast
from pathlib import Path

import libcst as cst
from libcst import MetadataWrapper, metadata

from codeintel.config.datasets import CallGraphEdgeRow
from codeintel.graphs.callgraph.resolution import (
    EdgeResolutionContext,
    build_evidence,
    resolve_callee,
    resolve_via_scip,
)

# =============================================================================
# CST-based Collection (Primary Strategy)
# =============================================================================

FUNCTION_NODE_TYPES = (cst.FunctionDef, getattr(cst, "AsyncFunctionDef", cst.FunctionDef))


class _FileCallGraphVisitor(cst.CSTVisitor):
    """LibCST visitor for collecting call graph edges with position metadata."""

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, rel_path: str, context: EdgeResolutionContext) -> None:
        self.rel_path = rel_path
        self.context = context
        self.current_function_goid: int | None = None
        self.edges: list[CallGraphEdgeRow] = []

    def _pos(self, node: cst.CSTNode) -> tuple[metadata.CodePosition, metadata.CodePosition] | None:
        try:
            pos = self.get_metadata(metadata.PositionProvider, node)
        except KeyError:
            return None
        if not isinstance(pos, metadata.CodeRange):
            return None
        return pos.start, pos.end

    def visit(self, node: cst.CSTNode) -> bool:
        """
        Visit a CST node, tracking function context and collecting call edges.

        Returns
        -------
        bool
            True to continue traversal.
        """
        if isinstance(node, FUNCTION_NODE_TYPES):
            span = self._pos(node)
            if span is None:
                return True
            start, end = span
            self.current_function_goid = self.context.function_index.lookup(
                self.rel_path, start.line, end.line
            )
            return True

        if isinstance(node, cst.Call):
            self._handle_call(node)
        return True

    def leave(self, node: cst.CSTNode) -> None:
        """Leave a CST node, clearing function context when exiting functions."""
        if isinstance(node, FUNCTION_NODE_TYPES):
            self.current_function_goid = None

    def _handle_call(self, node: cst.Call) -> None:
        if self.current_function_goid is None:
            spans = self.context.function_index.spans_for_path(self.rel_path)
            if spans:
                self.current_function_goid = spans[0].goid
        if self.current_function_goid is None:
            return

        span = self._pos(node)
        if span is None:
            return
        start, _end = span

        callee_name, attr_chain = extract_callee_cst(node.func)
        resolution = resolve_callee(
            callee_name,
            attr_chain,
            self.context.local_callees,
            self.context.global_callees,
            self.context.import_aliases,
        )
        scip_paths = self.context.scip_candidates_by_use_path.get(self.rel_path)
        if resolution.callee_goid is None and scip_paths:
            resolution = resolve_via_scip(scip_paths, self.context.def_goids_by_path)

        evidence = build_evidence(callee_name, attr_chain, resolution, scip_paths)
        self.edges.append(
            CallGraphEdgeRow(
                repo=self.context.repo,
                commit=self.context.commit,
                caller_goid_h128=self.current_function_goid,
                callee_goid_h128=resolution.callee_goid,
                callsite_path=self.rel_path,
                callsite_line=start.line,
                callsite_col=start.column,
                language="python",
                kind="direct" if resolution.callee_goid is not None else "unresolved",
                resolved_via=resolution.resolved_via,
                confidence=resolution.confidence,
                evidence_json=evidence,
            )
        )


def extract_callee_cst(expr: cst.CSTNode) -> tuple[str, list[str]]:
    """
    Extract callee name and attribute chain from a CST expression.

    Parameters
    ----------
    expr : cst.CSTNode
        The callee expression from a Call node.

    Returns
    -------
    tuple[str, list[str]]
        Base callee name and full attribute chain.
    """
    if isinstance(expr, cst.Name):
        return expr.value, [expr.value]
    if isinstance(expr, cst.Attribute):
        names: list[str] = []
        attr_node = expr
        while isinstance(attr_node, cst.Attribute):
            names.append(attr_node.attr.value)
            base = attr_node.value
            if isinstance(base, cst.Attribute):
                attr_node = base
                continue
            if isinstance(base, cst.Name):
                names.append(base.value)
            break
        names.reverse()
        return names[-1], names
    return "", []


def collect_edges_cst(
    rel_path: str,
    module: cst.Module,
    context: EdgeResolutionContext,
) -> list[CallGraphEdgeRow]:
    """
    Collect call edges via LibCST for a single module.

    This is the primary collection strategy, using LibCST's metadata
    infrastructure for accurate position tracking.

    Parameters
    ----------
    rel_path : str
        Relative path of the source file.
    module : cst.Module
        Parsed LibCST module.
    context : EdgeResolutionContext
        Resolution context with function index and callee maps.

    Returns
    -------
    list[CallGraphEdgeRow]
        Collected edges for the file.
    """
    visitor = _FileCallGraphVisitor(rel_path=rel_path, context=context)
    wrapper = MetadataWrapper(module)
    wrapper.resolve(metadata.PositionProvider)
    wrapper.visit(visitor)
    return visitor.edges


# =============================================================================
# AST-based Collection (Fallback Strategy)
# =============================================================================


def extract_callee_ast(expr: ast.AST) -> tuple[str, list[str]]:
    """
    Extract callee name and attribute chain from an AST expression.

    Parameters
    ----------
    expr : ast.AST
        The callee expression from a Call node.

    Returns
    -------
    tuple[str, list[str]]
        Base callee name and full attribute chain.
    """
    if isinstance(expr, ast.Name):
        return expr.id, [expr.id]
    if isinstance(expr, ast.Attribute):
        names: list[str] = []
        current: ast.AST = expr
        while isinstance(current, ast.Attribute):
            names.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            names.append(current.id)
            names.reverse()
            return names[0], names
    return "", []


class _AstCallGraphVisitor(ast.NodeVisitor):
    """AST visitor for collecting call graph edges as a fallback."""

    def __init__(self, rel_path: str, context: EdgeResolutionContext) -> None:
        self.rel_path = rel_path
        self.context = context
        self.current_goid: int | None = None
        self.edges: list[CallGraphEdgeRow] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Enter a function definition."""
        self._enter_function(node)
        self.generic_visit(node)
        self.current_goid = None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Enter an async function definition."""
        self._enter_function(node)
        self.generic_visit(node)
        self.current_goid = None

    def visit_Call(self, node: ast.Call) -> None:
        """Process a call expression."""
        if self.current_goid is None:
            return
        callee_name, attr_chain = extract_callee_ast(node.func)
        resolution = resolve_callee(
            callee_name,
            attr_chain,
            self.context.local_callees,
            self.context.global_callees,
            self.context.import_aliases,
        )
        scip_paths = self.context.scip_candidates_by_use_path.get(self.rel_path)
        if resolution.callee_goid is None and scip_paths:
            resolution = resolve_via_scip(scip_paths, self.context.def_goids_by_path)
        evidence = build_evidence(callee_name, attr_chain, resolution, scip_paths)
        self.edges.append(
            CallGraphEdgeRow(
                repo=self.context.repo,
                commit=self.context.commit,
                caller_goid_h128=self.current_goid,
                callee_goid_h128=resolution.callee_goid,
                callsite_path=self.rel_path,
                callsite_line=getattr(node, "lineno", 0),
                callsite_col=getattr(node, "col_offset", 0),
                language="python",
                kind="direct" if resolution.callee_goid is not None else "unresolved",
                resolved_via=resolution.resolved_via,
                confidence=resolution.confidence,
                evidence_json=evidence,
            )
        )
        self.generic_visit(node)

    def _enter_function(self, node: ast.AST) -> None:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None:
            return
        self.current_goid = self.context.function_index.lookup(
            self.rel_path,
            int(start),
            int(end) if end is not None else int(start),
        )


def collect_edges_ast(
    rel_path: str,
    file_path: Path,
    context: EdgeResolutionContext,
) -> list[CallGraphEdgeRow]:
    """
    Fallback AST-based call collection when LibCST metadata is unavailable.

    This strategy is used when CST-based collection fails or is not applicable.

    Parameters
    ----------
    rel_path : str
        Relative path of the source file.
    file_path : Path
        Absolute path to the source file.
    context : EdgeResolutionContext
        Resolution context with function index and callee maps.

    Returns
    -------
    list[CallGraphEdgeRow]
        Collected edges for the file.
    """
    source = file_path.read_text(encoding="utf8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    visitor = _AstCallGraphVisitor(rel_path, context)
    visitor.visit(tree)
    return visitor.edges


__all__ = [
    "collect_edges_ast",
    "collect_edges_cst",
    "extract_callee_ast",
    "extract_callee_cst",
]
