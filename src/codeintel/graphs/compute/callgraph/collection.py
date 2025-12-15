"""Call graph edge collection.

This module provides CST and AST-based visitors for collecting call graph
edges from Python source files.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, cast

import libcst as cst
from libcst import MetadataWrapper, metadata

from codeintel.core.schemas.generated_rows.graph import (
    GraphCallGraphEdgesRow as CallGraphEdgeRow,
)
from codeintel.graphs.compute.callgraph.persistence import (
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
    ResolutionResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from codeintel.graphs.compute.callgraph.types import (
        EdgeResolutionContext,
        ResolutionContext,
    )
    from codeintel.graphs.ports.parsing import ParsedModule


class LocalTypeTracker:
    """Track variable types from class instantiations.

    This class maintains a mapping of local variable names to their inferred
    class types based on assignment statements. It enables resolution of
    instance method calls like `obj.method()` where `obj = ClassName()`.

    Attributes
    ----------
    _variable_types
        Mapping of variable name to fully qualified class name.
    """

    def __init__(self) -> None:
        """Initialize an empty type tracker."""
        self._variable_types: dict[str, str] = {}

    def record_instantiation(
        self,
        var_name: str,
        class_name: str,
        import_aliases: Mapping[str, str],
    ) -> None:
        """Record a class instantiation assignment.

        Parameters
        ----------
        var_name
            The variable being assigned.
        class_name
            The class name from the instantiation expression.
        import_aliases
            Import alias mapping to resolve class to fully qualified name.
        """
        resolved = import_aliases.get(class_name, class_name)
        self._variable_types[var_name] = resolved

    def get_type(self, var_name: str) -> str | None:
        """Get the type of a variable if known.

        Parameters
        ----------
        var_name
            Variable name to look up.

        Returns
        -------
        str | None
            Fully qualified class name or None if unknown.
        """
        return self._variable_types.get(var_name)

    def clear(self) -> None:
        """Clear all tracked types (for entering new function scope)."""
        self._variable_types.clear()


def extract_class_name_from_call(func: cst.BaseExpression) -> str | None:
    """Extract class name from a Call node's func (for instantiation).

    Parameters
    ----------
    func
        The func attribute of a Call node.

    Returns
    -------
    str | None
        Class name if the call is a simple class instantiation, else None.
    """
    if isinstance(func, cst.Name):
        return func.value
    if isinstance(func, cst.Attribute):
        parts: list[str] = []
        current: cst.BaseExpression = func
        while isinstance(current, cst.Attribute):
            parts.append(current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            parts.append(current.value)
            parts.reverse()
            return ".".join(parts)
    return None


FUNCTION_NODE_TYPES = (cst.FunctionDef, getattr(cst, "AsyncFunctionDef", cst.FunctionDef))


_INSTANCE_METHOD_CHAIN_LENGTH = 2


class _FileCallGraphVisitor(cst.CSTVisitor):
    """LibCST visitor for collecting call graph edges with position metadata.

    This visitor tracks:
    - Function context (current function GOID)
    - Local variable types (from class instantiations)
    - Call expressions to build call graph edges
    """

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, rel_path: str, context: EdgeResolutionContext) -> None:
        self.rel_path = rel_path
        self.context = context
        self.current_function_goid: int | None = None
        self.edges: list[CallGraphEdgeRow] = []
        self.type_tracker = LocalTypeTracker()

    def _pos(self, node: cst.CSTNode) -> tuple[metadata.CodePosition, metadata.CodePosition] | None:
        try:
            pos = self.get_metadata(metadata.PositionProvider, node)
        except KeyError:
            return None
        if not isinstance(pos, metadata.CodeRange):
            return None
        return pos.start, pos.end

    def on_visit(self, node: cst.CSTNode) -> bool:
        """Visit a CST node, tracking function context, types, and calls.

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

            self.type_tracker.clear()
            return True

        if isinstance(node, (cst.Assign, cst.AnnAssign)):
            self._track_assignment(node)

        if isinstance(node, cst.Call):
            self._handle_call(node)
        return True

    def on_leave(self, original_node: cst.CSTNode) -> None:
        """Leave a CST node, clearing function context when exiting functions."""
        if isinstance(original_node, FUNCTION_NODE_TYPES):
            self.current_function_goid = None
            self.type_tracker.clear()

    def _track_assignment(self, node: cst.Assign | cst.AnnAssign) -> None:
        """Track variable type from class instantiation assignments.

        Parameters
        ----------
        node
            Assignment node to analyze.
        """
        value: cst.BaseExpression | None = None
        var_name: str | None = None

        if isinstance(node, cst.AnnAssign) and node.value is not None:
            value = node.value
            if isinstance(node.target, cst.Name):
                var_name = node.target.value
        elif isinstance(node, cst.Assign) and len(node.targets) == 1:
            target = node.targets[0].target
            if isinstance(target, cst.Name):
                var_name = target.value
                value = node.value

        if var_name is None or value is None:
            return

        if isinstance(value, cst.Call):
            class_name = extract_class_name_from_call(value.func)
            if class_name:
                self.type_tracker.record_instantiation(
                    var_name,
                    class_name,
                    self.context.import_aliases,
                )

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

        resolution: ResolutionResult | None = None
        if len(attr_chain) == _INSTANCE_METHOD_CHAIN_LENGTH:
            resolution = self._try_instance_method_resolution(attr_chain)

        if resolution is None or resolution.callee_goid is None:
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

    def _try_instance_method_resolution(
        self,
        attr_chain: list[str],
    ) -> ResolutionResult | None:
        """Attempt to resolve instance method calls via type tracking.

        For calls like `obj.method()` where we've tracked `obj = ClassName()`,
        attempt to resolve to `ClassName.method`.

        Parameters
        ----------
        attr_chain
            Two-element list [object_name, method_name].

        Returns
        -------
        ResolutionResult | None
            Resolution result if successful, None otherwise.
        """
        if len(attr_chain) != _INSTANCE_METHOD_CHAIN_LENGTH:
            return None

        obj_name, method_name = attr_chain
        class_type = self.type_tracker.get_type(obj_name)
        if class_type is None:
            return None

        method_qualname = f"{class_type}.{method_name}"
        goid = self.context.global_callees.get(method_qualname)
        if goid is not None:
            return ResolutionResult(
                callee_goid=goid,
                resolved_via="instance_method",
                confidence=0.75,
            )

        class_short_name = class_type.rsplit(".", 1)[-1]
        short_method_qualname = f"{class_short_name}.{method_name}"
        goid = self.context.global_callees.get(short_method_qualname)
        if goid is not None:
            return ResolutionResult(
                callee_goid=goid,
                resolved_via="instance_method",
                confidence=0.7,
            )

        for qualname, qgoid in self.context.global_callees.items():
            if qualname.endswith(f".{class_short_name}.{method_name}"):
                return ResolutionResult(
                    callee_goid=qgoid,
                    resolved_via="instance_method",
                    confidence=0.65,
                )

        return None


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
    "extract_class_name_from_call",
]
