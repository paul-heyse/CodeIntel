"""Pure call graph computation functions.

This module provides stateless functions for collecting and resolving
call graph edges without any database or file I/O.

Consolidated from:
- callgraph/resolution.py: Resolution logic and import alias handling
- callgraph/collectors.py: AST and CST-based edge collection

Architecture Notes
------------------
This module contains pure computation functions. For persistence, use
``adapters.callgraph_persistence``.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import libcst as cst
from libcst import MetadataWrapper, helpers, metadata

from codeintel.config.datasets import CallGraphEdgeRow
from codeintel.ingestion.infrastructure_utilities.paths import normalize_rel_path

if TYPE_CHECKING:
    from codeintel.graphs.catalog import FunctionSpanIndex
    from codeintel.graphs.ports.catalog import FunctionSpanData
    from codeintel.graphs.ports.parsing import ParsedModule


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class CallEdge:
    """Represents a call graph edge.

    Attributes
    ----------
    caller_goid
        GOID of the calling function.
    callee_goid
        GOID of the called function (None if unresolved).
    callee_name
        Name of the called function.
    call_line
        Line number of the call.
    rel_path
        Relative file path where call occurs.
    evidence
        Evidence supporting the edge (local, import, global, scip).
    confidence
        Confidence score (0.0 to 1.0).
    """

    caller_goid: int
    callee_goid: int | None
    callee_name: str
    call_line: int
    rel_path: str
    evidence: str
    confidence: float


@dataclass(frozen=True)
class ResolutionResult:
    """Structured outcome for a single callee resolution attempt.

    Attributes
    ----------
    callee_goid
        Resolved GOID or None if unresolved.
    resolved_via
        How the resolution was achieved.
    confidence
        Confidence score.
    """

    callee_goid: int | None
    resolved_via: str
    confidence: float


@dataclass
class ResolutionContext:
    """Context for call resolution operations.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    local_callees
        Local name to GOID mapping for current file.
    global_callees
        Global name to GOID mapping across repository.
    import_aliases
        Import alias to module mapping.
    scip_candidates
        SCIP-derived candidates by use path.
    def_goids_by_path
        Definition GOIDs by file path.
    """

    repo: str
    commit: str
    local_callees: Mapping[str, int] = field(default_factory=dict)
    global_callees: Mapping[str, int] = field(default_factory=dict)
    import_aliases: Mapping[str, str] = field(default_factory=dict)
    scip_candidates: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    def_goids_by_path: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeResolutionContext:
    """Resolution helpers shared across call graph visitors.

    This context provides all the mappings needed for resolving callees
    during edge collection traversals.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    function_index
        Index for looking up function spans.
    local_callees
        Local name to GOID mapping.
    global_callees
        Global name to GOID mapping.
    import_aliases
        Import alias to module mapping.
    scip_candidates_by_use_path
        SCIP candidates indexed by use path.
    def_goids_by_path
        Definition GOIDs indexed by path.
    """

    repo: str
    commit: str
    function_index: FunctionSpanIndex
    local_callees: dict[str, int]
    global_callees: dict[str, int]
    import_aliases: dict[str, str]
    scip_candidates_by_use_path: dict[str, tuple[str, ...]]
    def_goids_by_path: dict[str, int]


# =============================================================================
# Resolution Functions
# =============================================================================


def resolve_callee(
    callee_name: str,
    attr_chain: Sequence[str],
    local_callees: Mapping[str, int],
    global_callees: Mapping[str, int],
    import_aliases: Mapping[str, str],
) -> ResolutionResult:
    """Resolve a callee GOID using local/global maps and import aliases.

    Resolution precedence: local name -> local attr -> import alias -> global name -> global attr.

    Parameters
    ----------
    callee_name
        Base callee name extracted from the call expression.
    attr_chain
        Attribute chain on the callee (e.g., ["module", "func"]).
    local_callees
        Mapping of locally defined callables to GOIDs.
    global_callees
        Mapping of repository-global callables to GOIDs.
    import_aliases
        Import alias mapping from local alias to fully qualified module path.

    Returns
    -------
    ResolutionResult
        Structured resolution outcome with callee GOID, provenance, and confidence.

    Examples
    --------
    >>> local = {"func": 123}
    >>> result = resolve_callee("func", [], local, {}, {})
    >>> result.callee_goid
    123
    >>> result.resolved_via
    'local_name'
    """
    goid: int | None = None
    resolved_via = "unresolved"
    confidence = 0.0

    if callee_name in local_callees:
        goid = local_callees[callee_name]
        resolved_via = "local_name"
        confidence = 0.8
    elif attr_chain:
        joined = ".".join(attr_chain)
        goid = local_callees.get(joined) or local_callees.get(attr_chain[-1])
        if goid is not None:
            resolved_via = "local_attr"
            confidence = 0.75
        else:
            root = attr_chain[0]
            alias_target = import_aliases.get(root)
            if alias_target:
                qualified = (
                    alias_target if len(attr_chain) == 1 else ".".join([alias_target, *attr_chain[1:]])
                )
                goid = local_callees.get(qualified) or global_callees.get(qualified)
                if goid is not None:
                    resolved_via = "import_alias"
                    confidence = 0.7

    if goid is None and callee_name in global_callees:
        goid = global_callees[callee_name]
        resolved_via = "global_name"
        confidence = 0.6
    elif goid is None and attr_chain:
        qualified = ".".join(attr_chain)
        goid = global_callees.get(qualified) or global_callees.get(attr_chain[-1])
        if goid is not None:
            resolved_via = "global_name"
            confidence = 0.6

    return ResolutionResult(callee_goid=goid, resolved_via=resolved_via, confidence=confidence)


def resolve_via_scip(
    candidate_def_paths: tuple[str, ...],
    def_goids_by_path: Mapping[str, int],
) -> ResolutionResult:
    """Resolve using SCIP definition paths when primary resolution fails.

    Parameters
    ----------
    candidate_def_paths
        Candidate definition paths produced by SCIP cross-references.
    def_goids_by_path
        Mapping from normalized definition paths to GOIDs.

    Returns
    -------
    ResolutionResult
        Resolution outcome using SCIP data or unresolved when none match.
    """
    for def_path in candidate_def_paths:
        goid = def_goids_by_path.get(normalize_rel_path(def_path))
        if goid is not None:
            return ResolutionResult(callee_goid=goid, resolved_via="scip_def_path", confidence=0.55)
    return ResolutionResult(callee_goid=None, resolved_via="unresolved", confidence=0.0)


def build_evidence(
    callee_name: str,
    attr_chain: Sequence[str],
    resolution: ResolutionResult,
    scip_candidates: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Construct evidence payload in a consistent shape.

    Parameters
    ----------
    callee_name
        Base callee name extracted from the call expression.
    attr_chain
        Attribute chain on the callee; empty when no attributes are present.
    resolution
        Resolution outcome detailing GOID and provenance.
    scip_candidates
        SCIP candidate definition paths if available.

    Returns
    -------
    dict[str, object]
        Evidence payload suitable for persistence or debugging.
    """
    evidence: dict[str, object] = {
        "callee_name": callee_name,
        "attr_chain": list(attr_chain) if attr_chain else None,
        "resolved_via": resolution.resolved_via,
    }
    if scip_candidates:
        evidence["scip_candidates"] = list(scip_candidates)
    return evidence


def build_callee_map(spans: Sequence[FunctionSpanData]) -> dict[str, int]:
    """Build a global name to GOID mapping from function spans.

    Parameters
    ----------
    spans
        Function spans to index.

    Returns
    -------
    dict[str, int]
        Mapping from function name (local and qualified) to GOID.
    """
    mapping: dict[str, int] = {}
    for span in spans:
        mapping.setdefault(span.qualname, span.goid)
        mapping.setdefault(span.local_name, span.goid)
    return mapping


# =============================================================================
# Import Alias Resolution
# =============================================================================


def _attr_to_str(node: cst.CSTNode) -> str:
    """Render a LibCST Name/Attribute into a dotted string.

    Returns
    -------
    str
        Dotted representation or empty string when it cannot be resolved.
    """
    full_name = helpers.get_full_name_for_node(node)
    if full_name:
        return full_name
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        parts: list[str] = []
        cur: cst.BaseExpression | cst.Name = node
        while isinstance(cur, cst.Attribute):
            parts.append(cur.attr.value)
            cur = cur.value
        if isinstance(cur, cst.Name):
            parts.append(cur.value)
        parts.reverse()
        return ".".join(parts) if parts else ""
    return ""


def _resolve_base_module(current_module: str, node: cst.ImportFrom) -> str | None:
    """Resolve the base module for an ImportFrom, handling relative imports.

    Returns
    -------
    str | None
        Dotted base module or None when it cannot be resolved.
    """
    module_part = _attr_to_str(node.module) if node.module is not None else ""
    level = len(node.relative) if node.relative else 0

    if level == 0:
        return module_part or None

    package_parts = current_module.split(".")
    if package_parts:
        package_parts = package_parts[:-1]  # drop module name

    if level > 1:
        package_parts = package_parts[: max(0, len(package_parts) - (level - 1))]

    if level >= 1 and not package_parts:
        return None

    if module_part:
        package_parts.append(module_part)

    if not package_parts:
        return None
    return ".".join(package_parts)


def _record_import_aliases(node: cst.Import, alias_map: dict[str, str]) -> None:
    """Populate alias_map with aliases from an Import statement."""
    for alias in node.names:
        target = _attr_to_str(cast("cst.CSTNode", alias.name))
        asname_node = alias.asname.name if alias.asname else None
        asname = (
            _attr_to_str(cast("cst.CSTNode", asname_node))
            if asname_node is not None
            else target.split(".")[-1]
        )
        if target:
            alias_map[asname] = target


def _record_import_from_aliases(
    node: cst.ImportFrom,
    alias_map: dict[str, str],
    current_module: str | None = None,
) -> None:
    """Populate alias_map with aliases from an ImportFrom statement."""
    base_module: str | None
    if current_module is None:
        if node.module is None:
            return
        module_name = _attr_to_str(node.module)
        base_module = module_name or None
    else:
        base_module = _resolve_base_module(current_module, node)
    if not base_module:
        return
    names = node.names
    if isinstance(names, cst.ImportStar):
        return
    for alias in cast("list[cst.ImportAlias]", names):
        target = f"{base_module}.{_attr_to_str(cast('cst.CSTNode', alias.name))}"
        asname_node = alias.asname.name if alias.asname else None
        asname = (
            _attr_to_str(cast("cst.CSTNode", asname_node))
            if asname_node is not None
            else _attr_to_str(cast("cst.CSTNode", alias.name))
        )
        alias_map[asname] = target


def collect_aliases(module: cst.Module, current_module: str | None = None) -> dict[str, str]:
    """Collect import aliases for a module.

    Parameters
    ----------
    module
        Parsed LibCST module.
    current_module
        Fully qualified name of current module (for relative imports).

    Returns
    -------
    dict[str, str]
        Mapping of alias -> fully qualified import target.
    """
    aliases: dict[str, str] = {}

    class _AliasVisitor(cst.CSTVisitor):
        def __init__(self, alias_map: dict[str, str], module_name: str | None) -> None:
            self.alias_map = alias_map
            self.module_name = module_name

        def on_visit(self, node: cst.CSTNode) -> bool:
            if isinstance(node, cst.Import):
                _record_import_aliases(node, self.alias_map)
            elif isinstance(node, cst.ImportFrom):
                _record_import_from_aliases(node, self.alias_map, self.module_name)
            return True

    module.visit(_AliasVisitor(aliases, current_module))
    return aliases


def collect_import_edges(current_module: str, module: cst.Module) -> set[tuple[str, str]]:
    """Collect import edges (src_module, dst_module) for a given CST module.

    Parameters
    ----------
    current_module
        Fully qualified module name of the file being parsed.
    module
        Parsed LibCST module.

    Returns
    -------
    set[tuple[str, str]]
        Edges from current_module to imported modules.
    """
    edges: set[tuple[str, str]] = set()
    _collect_imports(current_module, module, edges)
    return edges


def _collect_imports(current_module: str, module: cst.Module, edges: set[tuple[str, str]]) -> None:
    """Populate edges set with imports discovered in the module."""

    class _ImportVisitor(cst.CSTVisitor):
        def __init__(self, current: str, edge_set: set[tuple[str, str]]) -> None:
            self.current = current
            self.edge_set = edge_set

        def on_visit(self, node: cst.CSTNode) -> bool:
            if isinstance(node, cst.Import):
                handle_import(node, self.current, self.edge_set)
            elif isinstance(node, cst.ImportFrom):
                handle_import_from(node, self.current, self.edge_set)
            return True

    module.visit(_ImportVisitor(current_module, edges))


def handle_import(node: cst.Import, current_module: str, edges: set[tuple[str, str]]) -> None:
    """Handle standard import statements."""
    for name in node.names:
        module_str = _attr_to_str(name.name)
        if module_str:
            edges.add((current_module, module_str))


def handle_import_from(
    node: cst.ImportFrom,
    current_module: str,
    edges: set[tuple[str, str]],
) -> None:
    """Handle import-from statements including relative imports."""
    base = _resolve_base_module(current_module, node)
    if base is None:
        return
    edges.add((current_module, base))
    names = node.names
    if isinstance(names, cst.ImportStar):
        return
    if node.module is None:
        for alias in cast("list[cst.ImportAlias]", names):
            target = _attr_to_str(cast("cst.CSTNode", alias.name))
            if target:
                edges.add((current_module, f"{base}.{target}"))


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

    # Handle CallGraphEdgeRow type
    from codeintel.graphs.adapters.callgraph_persistence import (  # noqa: PLC0415
        dedupe_edge_rows,
        default_edge_key,
    )

    return dedupe_edge_rows(cast("list[CallGraphEdgeRow]", list(edges)), key_fn or default_edge_key)


__all__ = [
    "CallEdge",
    "EdgeResolutionContext",
    "ResolutionContext",
    "ResolutionResult",
    "build_callee_map",
    "build_evidence",
    "collect_aliases",
    "collect_call_sites",
    "collect_edges_ast",
    "collect_edges_cst",
    "collect_edges_for_function",
    "collect_import_edges",
    "dedupe_edges",
    "extract_callee_ast",
    "extract_callee_cst",
    "handle_import",
    "handle_import_from",
    "resolve_callee",
    "resolve_via_scip",
]
