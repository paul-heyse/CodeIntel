"""Call graph edge collection.

This module provides CST and AST-based visitors for collecting call graph
edges from Python source files.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import libcst as cst
from libcst import MetadataWrapper, metadata

from codeintel.config.datasets import CallGraphEdgeRow
from codeintel.graphs.adapters.callgraph_persistence import (
    dedupe_edge_rows,
    default_edge_key,
)
from codeintel.graphs.compute.callgraph.resolution import (
    build_evidence,
    resolve_callee,
    resolve_via_scip,
)
from codeintel.graphs.compute.callgraph.types import (
    CallEdge,
    EdgeResolutionContext,
    ResolutionContext,
)

if TYPE_CHECKING:
    from codeintel.graphs.ports.parsing import ParsedModule


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
        """Visit a CST node, tracking function context and collecting call edges.

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
    """Extract callee name and attribute chain from a CST expression.

    Parameters
    ----------
    expr
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
    """Collect call edges via LibCST for a single module.

    This is the primary collection strategy, using LibCST's metadata
    infrastructure for accurate position tracking.

    Parameters
    ----------
    rel_path
        Relative path of the source file.
    module
        Parsed LibCST module.
    context
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
    """Extract callee name and attribute chain from an AST expression.

    Parameters
    ----------
    expr
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
    """Fallback AST-based call collection when LibCST metadata is unavailable.

    This strategy is used when CST-based collection fails or is not applicable.

    Parameters
    ----------
    rel_path
        Relative path of the source file.
    file_path
        Absolute path to the source file.
    context
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


# =============================================================================
# Edge Utilities
# =============================================================================


def collect_call_sites(
    module: ParsedModule,
    function_span: tuple[int, int],
) -> Sequence[tuple[str, Sequence[str], int]]:
    """Extract call sites from a parsed module within a function span.

    Parameters
    ----------
    module
        Parsed module containing the function.
    function_span
        (start_line, end_line) of the function.

    Returns
    -------
    Sequence[tuple[str, Sequence[str], int]]
        Sequence of (callee_name, attr_chain, line_number) tuples.
    """
    if module.ast_module is None:
        return []

    start_line, end_line = function_span
    call_sites: list[tuple[str, Sequence[str], int]] = []

    for node in ast.walk(module.ast_module):
        if isinstance(node, ast.Call):
            line = getattr(node, "lineno", 0)
            if start_line <= line <= end_line:
                callee_name, attr_chain = _extract_callee_info(node.func)
                if callee_name:
                    call_sites.append((callee_name, attr_chain, line))

    return call_sites


def _extract_callee_info(node: object) -> tuple[str | None, list[str]]:
    """Extract callee name and attribute chain from a call target.

    Parameters
    ----------
    node
        AST node representing the call target.

    Returns
    -------
    tuple[str | None, list[str]]
        (base_name, attribute_chain) tuple.
    """
    if isinstance(node, ast.Name):
        return node.id, []
    if isinstance(node, ast.Attribute):
        chain: list[str] = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            chain.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            return current.id, chain
    return None, []


def collect_edges_for_function(
    caller_goid: int,
    rel_path: str,
    module: ParsedModule,
    function_span: tuple[int, int],
    context: ResolutionContext,
) -> list[CallEdge]:
    """Collect call edges for a single function.

    Parameters
    ----------
    caller_goid
        GOID of the calling function.
    rel_path
        Relative file path.
    module
        Parsed module containing the function.
    function_span
        (start_line, end_line) of the function.
    context
        Resolution context with scope mappings.

    Returns
    -------
    list[CallEdge]
        Collected call edges.
    """
    edges: list[CallEdge] = []
    call_sites = collect_call_sites(module, function_span)

    for callee_name, attr_chain, line in call_sites:
        resolution = resolve_callee(
            callee_name,
            attr_chain,
            context.local_callees,
            context.global_callees,
            context.import_aliases,
        )
        edges.append(
            CallEdge(
                caller_goid=caller_goid,
                callee_goid=resolution.callee_goid,
                callee_name=callee_name,
                call_line=line,
                rel_path=rel_path,
                evidence=resolution.resolved_via,
                confidence=resolution.confidence,
            )
        )

    return edges


def dedupe_edges(
    edges: Sequence[CallEdge] | list[CallGraphEdgeRow],
    key_fn: Callable[[CallGraphEdgeRow], tuple[object, ...]] | None = None,
) -> list[CallEdge] | list[CallGraphEdgeRow]:
    """Remove duplicate edges, keeping highest confidence.

    Parameters
    ----------
    edges
        Edges to deduplicate.
    key_fn
        Optional key function for CallGraphEdgeRow edges.

    Returns
    -------
    list[CallEdge] | list[CallGraphEdgeRow]
        Deduplicated edges.
    """
    if not edges:
        return []

    # Handle CallEdge type
    if isinstance(edges[0], CallEdge):
        seen: dict[tuple[int, int | None, int], CallEdge] = {}
        for edge in edges:
            edge = cast("CallEdge", edge)
            key = (edge.caller_goid, edge.callee_goid, edge.call_line)
            existing = seen.get(key)
            if existing is None or edge.confidence > existing.confidence:
                seen[key] = edge
        return list(seen.values())

    return dedupe_edge_rows(cast("list[CallGraphEdgeRow]", list(edges)), key_fn or default_edge_key)


__all__ = [
    "collect_call_sites",
    "collect_edges_ast",
    "collect_edges_cst",
    "collect_edges_for_function",
    "dedupe_edges",
    "extract_callee_ast",
    "extract_callee_cst",
]
