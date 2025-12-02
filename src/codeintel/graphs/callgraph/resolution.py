"""Call graph resolution logic, context structures, and import alias handling.

This module consolidates:
- ResolutionResult and resolve_callee from call_resolution.py
- EdgeResolutionContext from call_context.py
- Import alias collection from import_resolver.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import libcst as cst
from libcst import helpers

from codeintel.ingestion.infrastructure_utilities.paths import normalize_rel_path

if TYPE_CHECKING:
    from codeintel.graphs.catalog import FunctionSpanIndex


# =============================================================================
# Resolution Result (from call_resolution.py)
# =============================================================================


@dataclass(frozen=True)
class ResolutionResult:
    """Structured outcome for a single callee resolution attempt."""

    callee_goid: int | None
    resolved_via: str
    confidence: float


def resolve_callee(
    callee_name: str,
    attr_chain: list[str],
    local_callees: dict[str, int],
    global_callees: dict[str, int],
    import_aliases: dict[str, str],
) -> ResolutionResult:
    """
    Resolve a callee GOID using local/global maps and import aliases.

    Resolution precedence: local name -> local attr -> import alias -> global name -> global attr.

    Parameters
    ----------
    callee_name : str
        Base callee name extracted from the call expression.
    attr_chain : list[str]
        Attribute chain on the callee (e.g., ["module", "func"]).
    local_callees : dict[str, int]
        Mapping of locally defined callables to GOIDs.
    global_callees : dict[str, int]
        Mapping of repository-global callables to GOIDs.
    import_aliases : dict[str, str]
        Import alias mapping from local alias to fully qualified module path.

    Returns
    -------
    ResolutionResult
        Structured resolution outcome with callee GOID, provenance, and confidence.
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
                    alias_target
                    if len(attr_chain) == 1
                    else ".".join([alias_target, *attr_chain[1:]])
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
    candidate_def_paths: tuple[str, ...], def_goids_by_path: dict[str, int]
) -> ResolutionResult:
    """
    Resolve using SCIP definition paths when primary resolution fails.

    Parameters
    ----------
    candidate_def_paths : tuple[str, ...]
        Candidate definition paths produced by SCIP cross-references.
    def_goids_by_path : dict[str, int]
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
    attr_chain: list[str],
    resolution: ResolutionResult,
    scip_candidates: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """
    Construct evidence payload in a consistent shape.

    Parameters
    ----------
    callee_name : str
        Base callee name extracted from the call expression.
    attr_chain : list[str]
        Attribute chain on the callee; empty when no attributes are present.
    resolution : ResolutionResult
        Resolution outcome detailing GOID and provenance.
    scip_candidates : tuple[str, ...], optional
        SCIP candidate definition paths if available.

    Returns
    -------
    dict[str, object]
        Evidence payload suitable for persistence or debugging.
    """
    evidence: dict[str, object] = {
        "callee_name": callee_name,
        "attr_chain": attr_chain or None,
        "resolved_via": resolution.resolved_via,
    }
    if scip_candidates:
        evidence["scip_candidates"] = list(scip_candidates)
    return evidence


# =============================================================================
# Edge Resolution Context (from call_context.py)
# =============================================================================


@dataclass(frozen=True)
class EdgeResolutionContext:
    """Resolution helpers shared across call graph visitors."""

    repo: str
    commit: str
    function_index: FunctionSpanIndex
    local_callees: dict[str, int]
    global_callees: dict[str, int]
    import_aliases: dict[str, str]
    scip_candidates_by_use_path: dict[str, tuple[str, ...]]
    def_goids_by_path: dict[str, int]


# =============================================================================
# Import Alias Resolution (from import_resolver.py)
# =============================================================================


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
    node: cst.ImportFrom, alias_map: dict[str, str], current_module: str | None = None
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


def _attr_to_str(node: cst.CSTNode) -> str:
    """
    Render a LibCST Name/Attribute into a dotted string.

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


def collect_aliases(module: cst.Module, current_module: str | None = None) -> dict[str, str]:
    """
    Collect import aliases for a module.

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


def _resolve_base_module(current_module: str, node: cst.ImportFrom) -> str | None:
    """
    Resolve the base module for an ImportFrom, handling relative imports.

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


def collect_import_edges(current_module: str, module: cst.Module) -> set[tuple[str, str]]:
    """
    Collect import edges (src_module, dst_module) for a given CST module.

    Parameters
    ----------
    current_module:
        Fully qualified module name of the file being parsed.
    module:
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
    node: cst.ImportFrom, current_module: str, edges: set[tuple[str, str]]
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


__all__ = [
    "EdgeResolutionContext",
    "ResolutionResult",
    "build_evidence",
    "collect_aliases",
    "collect_import_edges",
    "handle_import",
    "handle_import_from",
    "resolve_callee",
    "resolve_via_scip",
]
