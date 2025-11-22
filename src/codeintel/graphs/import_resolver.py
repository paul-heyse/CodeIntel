"""Shared helpers for LibCST import and alias resolution."""

from __future__ import annotations

from typing import cast

import libcst as cst
from libcst import helpers


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


def _record_import_from_aliases(node: cst.ImportFrom, alias_map: dict[str, str]) -> None:
    """Populate alias_map with aliases from an ImportFrom statement."""
    if node.module is None:
        return
    module_name = _attr_to_str(node.module)
    if not module_name:
        return
    names = node.names
    if isinstance(names, cst.ImportStar):
        return
    for alias in cast("list[cst.ImportAlias]", names):
        target = f"{module_name}.{_attr_to_str(cast('cst.CSTNode', alias.name))}"
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


def collect_aliases(module: cst.Module) -> dict[str, str]:
    """
    Collect import aliases for a module.

    Returns
    -------
    dict[str, str]
        Mapping of alias -> fully qualified import target.
    """
    aliases: dict[str, str] = {}

    class _AliasVisitor(cst.CSTVisitor):
        def __init__(self, alias_map: dict[str, str]) -> None:
            self.alias_map = alias_map

        def on_visit(self, node: cst.CSTNode) -> bool:
            if isinstance(node, cst.Import):
                _record_import_aliases(node, self.alias_map)
            elif isinstance(node, cst.ImportFrom):
                _record_import_from_aliases(node, self.alias_map)
            return True

    module.visit(_AliasVisitor(aliases))
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

    class _ImportVisitor(cst.CSTVisitor):
        def __init__(self, edge_set: set[tuple[str, str]]) -> None:
            self.edge_set = edge_set

        def visit_Import(self, node: cst.Import) -> None:  # noqa: N802 - libcst visitor API
            for name in node.names:
                module_str = _attr_to_str(name.name)
                if module_str:
                    self.edge_set.add((current_module, module_str))

        def visit_ImportFrom(self, node: cst.ImportFrom) -> None:  # noqa: N802 - libcst visitor API
            base = _resolve_base_module(current_module, node)
            if base is None:
                return
            self.edge_set.add((current_module, base))

    module.visit(_ImportVisitor(edges))
    return edges


__all__ = ["collect_aliases", "collect_import_edges"]
