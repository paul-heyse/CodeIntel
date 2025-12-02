"""Pure symbol use analysis functions.

This module provides stateless functions for analyzing symbol definitions
and uses without any database or file I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolOccurrence:
    """Represents a symbol occurrence.

    Attributes
    ----------
    symbol
        Symbol identifier.
    rel_path
        Relative file path.
    line
        Line number.
    roles
        Symbol role bitmask.
    """

    symbol: str
    rel_path: str
    line: int
    roles: int

    @property
    def is_definition(self) -> bool:
        """Check if this is a definition occurrence.

        Returns
        -------
        bool
            True if definition bit is set.
        """
        return bool(self.roles & 1)

    @property
    def is_reference(self) -> bool:
        """Check if this is a reference occurrence.

        Returns
        -------
        bool
            True if any reference bit is set.
        """
        return bool(self.roles & (2 | 4 | 8))


@dataclass(frozen=True)
class SymbolUseEdge:
    """Represents a symbol definition-to-use edge.

    Attributes
    ----------
    symbol
        Symbol identifier.
    def_path
        Path where symbol is defined.
    use_path
        Path where symbol is used.
    same_file
        Whether definition and use are in same file.
    same_module
        Whether definition and use are in same module.
    def_goid
        Optional definition GOID.
    use_goid
        Optional use GOID.
    """

    symbol: str
    def_path: str
    use_path: str
    same_file: bool
    same_module: bool
    def_goid: int | None = None
    use_goid: int | None = None


@dataclass(frozen=True)
class SymbolUseRow:
    """Row data for graph.symbol_use_edges table.

    Attributes
    ----------
    symbol
        Symbol identifier.
    def_path
        Definition path.
    use_path
        Use path.
    same_file
        Whether same file.
    same_module
        Whether same module.
    def_goid_h128
        Definition GOID.
    use_goid_h128
        Use GOID.
    """

    symbol: str
    def_path: str
    use_path: str
    same_file: bool
    same_module: bool
    def_goid_h128: int | None
    use_goid_h128: int | None


def build_def_map(occurrences: Sequence[SymbolOccurrence]) -> dict[str, str]:
    """Build a symbol to definition path mapping.

    Parameters
    ----------
    occurrences
        Symbol occurrences to process.

    Returns
    -------
    dict[str, str]
        Symbol to definition path mapping.
    """
    def_map: dict[str, str] = {}
    for occ in occurrences:
        if occ.is_definition and occ.symbol not in def_map:
            def_map[occ.symbol] = occ.rel_path
    return def_map


def build_use_edges(
    occurrences: Sequence[SymbolOccurrence],
    def_map: Mapping[str, str],
    module_by_path: Mapping[str, str],
) -> list[SymbolUseEdge]:
    """Build symbol use edges from occurrences.

    Parameters
    ----------
    occurrences
        Symbol occurrences to process.
    def_map
        Symbol to definition path mapping.
    module_by_path
        Path to module name mapping.

    Returns
    -------
    list[SymbolUseEdge]
        Symbol use edges.
    """
    edges: list[SymbolUseEdge] = []
    seen: set[tuple[str, str, str]] = set()

    for occ in occurrences:
        if not occ.is_reference:
            continue

        def_path = def_map.get(occ.symbol)
        if not def_path:
            continue

        key = (occ.symbol, def_path, occ.rel_path)
        if key in seen:
            continue
        seen.add(key)

        same_file = def_path == occ.rel_path
        def_module = module_by_path.get(def_path)
        use_module = module_by_path.get(occ.rel_path)
        same_module = def_module is not None and def_module == use_module

        edges.append(
            SymbolUseEdge(
                symbol=occ.symbol,
                def_path=def_path,
                use_path=occ.rel_path,
                same_file=same_file,
                same_module=same_module,
            )
        )

    return edges


def build_use_def_mapping(
    occurrences: Sequence[SymbolOccurrence],
    def_map: Mapping[str, str],
) -> dict[str, set[str]]:
    """Build use path to definition paths mapping.

    Parameters
    ----------
    occurrences
        Symbol occurrences to process.
    def_map
        Symbol to definition path mapping.

    Returns
    -------
    dict[str, set[str]]
        Use path to set of definition paths.
    """
    mapping: dict[str, set[str]] = {}
    for occ in occurrences:
        if not occ.is_reference:
            continue
        def_path = def_map.get(occ.symbol)
        if not def_path:
            continue
        mapping.setdefault(occ.rel_path, set()).add(def_path)
    return mapping


def edges_to_rows(edges: Sequence[SymbolUseEdge]) -> list[SymbolUseRow]:
    """Convert symbol use edges to database rows.

    Parameters
    ----------
    edges
        Symbol use edges.

    Returns
    -------
    list[SymbolUseRow]
        Rows for persistence.
    """
    return [
        SymbolUseRow(
            symbol=edge.symbol,
            def_path=edge.def_path,
            use_path=edge.use_path,
            same_file=edge.same_file,
            same_module=edge.same_module,
            def_goid_h128=edge.def_goid,
            use_goid_h128=edge.use_goid,
        )
        for edge in edges
    ]


def parse_symbol_roles(roles_value: object) -> int:
    """Parse symbol roles from various input formats.

    Parameters
    ----------
    roles_value
        Roles value from SCIP data.

    Returns
    -------
    int
        Symbol roles bitmask.
    """
    if roles_value is None:
        return 0
    if isinstance(roles_value, int):
        return roles_value
    if isinstance(roles_value, str):
        try:
            return int(roles_value)
        except ValueError:
            return 0
    return 0


__all__ = [
    "SymbolOccurrence",
    "SymbolUseEdge",
    "SymbolUseRow",
    "build_def_map",
    "build_use_def_mapping",
    "build_use_edges",
    "edges_to_rows",
    "parse_symbol_roles",
]
