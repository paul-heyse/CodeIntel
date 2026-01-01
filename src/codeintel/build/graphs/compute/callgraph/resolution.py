"""Call graph resolution logic.

This module provides functions for resolving callees to GOIDs using
local/global maps, import aliases, and SCIP cross-references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import libcst as cst
from libcst import helpers

from codeintel.build.graphs.compute.callgraph.types import ResolutionResult
from codeintel.core.paths import normalize_path

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.core.catalog import FunctionSpan


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
        goid = def_goids_by_path.get(normalize_path(def_path))
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


def build_callee_map(spans: Sequence[FunctionSpan]) -> dict[str, int]:
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


def attr_to_str(node: cst.CSTNode) -> str:
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


def resolve_base_module(current_module: str, node: cst.ImportFrom) -> str | None:
    """Resolve the base module for an ImportFrom, handling relative imports.

    Returns
    -------
    str | None
        Dotted base module or None when it cannot be resolved.
    """
    module_part = attr_to_str(node.module) if node.module is not None else ""
    level = len(node.relative) if node.relative else 0

    if level == 0:
        return module_part or None

    package_parts = current_module.split(".")
    if package_parts:
        package_parts = package_parts[:-1]

    if level > 1:
        package_parts = package_parts[: max(0, len(package_parts) - (level - 1))]

    if level >= 1 and not package_parts:
        return None

    if module_part:
        package_parts.append(module_part)

    if not package_parts:
        return None
    return ".".join(package_parts)


def record_import_aliases(node: cst.Import, alias_map: dict[str, str]) -> None:
    """Populate alias_map with aliases from an Import statement."""
    for alias in node.names:
        target = attr_to_str(cast("cst.CSTNode", alias.name))
        asname_node = alias.asname.name if alias.asname else None
        asname = (
            attr_to_str(cast("cst.CSTNode", asname_node))
            if asname_node is not None
            else target.split(".")[-1]
        )
        if target:
            alias_map[asname] = target


def record_import_from_aliases(
    node: cst.ImportFrom,
    alias_map: dict[str, str],
    current_module: str | None = None,
) -> None:
    """Populate alias_map with aliases from an ImportFrom statement."""
    base_module: str | None
    if current_module is None:
        if node.module is None:
            return
        module_name = attr_to_str(node.module)
        base_module = module_name or None
    else:
        base_module = resolve_base_module(current_module, node)
    if not base_module:
        return
    names = node.names
    if isinstance(names, cst.ImportStar):
        return
    for alias in cast("list[cst.ImportAlias]", names):
        target = f"{base_module}.{attr_to_str(cast('cst.CSTNode', alias.name))}"
        asname_node = alias.asname.name if alias.asname else None
        asname = (
            attr_to_str(cast("cst.CSTNode", asname_node))
            if asname_node is not None
            else attr_to_str(cast("cst.CSTNode", alias.name))
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
                record_import_aliases(node, self.alias_map)
            elif isinstance(node, cst.ImportFrom):
                record_import_from_aliases(node, self.alias_map, self.module_name)
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
        module_str = attr_to_str(name.name)
        if module_str:
            edges.add((current_module, module_str))


def handle_import_from(
    node: cst.ImportFrom,
    current_module: str,
    edges: set[tuple[str, str]],
) -> None:
    """Handle import-from statements including relative imports."""
    base = resolve_base_module(current_module, node)
    if base is None:
        return
    edges.add((current_module, base))
    names = node.names
    if isinstance(names, cst.ImportStar):
        return
    if node.module is None:
        for alias in cast("list[cst.ImportAlias]", names):
            target = attr_to_str(cast("cst.CSTNode", alias.name))
            if target:
                edges.add((current_module, f"{base}.{target}"))


__all__ = [
    "attr_to_str",
    "build_callee_map",
    "build_evidence",
    "collect_aliases",
    "collect_import_edges",
    "handle_import",
    "handle_import_from",
    "record_import_aliases",
    "record_import_from_aliases",
    "resolve_base_module",
    "resolve_callee",
    "resolve_via_scip",
]
