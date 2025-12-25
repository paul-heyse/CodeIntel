"""Canonical graph row data models.

This module defines all graph-related row types as frozen dataclasses.
Each row type includes a `to_tuple()` method for DuckDB serialization,
with fields ordered to match the INSERT column order.

These are the canonical definitions - all other modules should import
from here rather than defining their own row types.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class CFGBlockRow:
    """Row data for graph.cfg_blocks table.

    Attributes
    ----------
    function_goid_h128
        Function GOID hash.
    block_idx
        Block index within function.
    block_id
        Block identifier string.
    label
        Human-readable block label.
    file_path
        Source file path.
    start_line
        Starting line number.
    end_line
        Ending line number.
    kind
        Block kind (entry, body, exit, loop_head, etc.).
    stmts_json
        Statements as JSON string.
    in_degree
        Number of incoming edges.
    out_degree
        Number of outgoing edges.
    """

    function_goid_h128: int
    block_idx: int
    block_id: str
    label: str
    file_path: str
    start_line: int
    end_line: int
    kind: str
    stmts_json: str | object
    in_degree: int
    out_degree: int

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class CFGEdgeRow:
    """Row data for graph.cfg_edges table.

    Attributes
    ----------
    function_goid_h128
        Function GOID hash.
    src_block_id
        Source block identifier.
    dst_block_id
        Destination block identifier.
    edge_kind
        Edge kind (fallthrough, true, false, etc.).
    """

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    edge_kind: str | None

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class DFGEdgeRow:
    """Row data for graph.dfg_edges table.

    Attributes
    ----------
    function_goid_h128
        Function GOID hash.
    src_block_id
        Source block identifier.
    dst_block_id
        Destination block identifier.
    src_var
        Source variable name.
    dst_var
        Destination variable name.
    edge_kind
        Edge kind descriptor.
    via_phi
        Whether edge passes through phi node.
    use_kind
        Use kind descriptor.
    """

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    src_var: str | None
    dst_var: str | None
    edge_kind: str | None
    via_phi: bool
    use_kind: str | None

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class ImportModuleRow:
    """Row data for graph.import_modules table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    module
        Module name.
    scc_id
        Strongly connected component ID.
    component_size
        Size of the SCC.
    layer
        Topological layer (None if in cycle).
    cycle_group
        Cycle group ID.
    """

    repo: str
    commit: str
    module: str
    scc_id: int
    component_size: int
    layer: int | None
    cycle_group: int

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class ImportEdgeRow:
    """Row data for graph.import_graph_edges table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    src_module
        Source module name.
    dst_module
        Destination module name.
    src_fan_out
        Fan-out of source module.
    dst_fan_in
        Fan-in of destination module.
    cycle_group
        Cycle group ID.
    module_layer
        Layer of the source module.
    """

    repo: str
    commit: str
    src_module: str
    dst_module: str
    src_fan_out: int
    dst_fan_in: int
    cycle_group: int
    module_layer: int | None

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class SymbolUseRow:
    """Row data for graph.symbol_use_edges table.

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
    def_goid_h128
        Definition GOID hash.
    use_goid_h128
        Use GOID hash.
    """

    symbol: str
    def_path: str
    use_path: str
    same_file: bool
    same_module: bool
    def_goid_h128: int | None
    use_goid_h128: int | None

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class GoidRow:
    """Row data for core.goids table.

    Attributes
    ----------
    goid_h128
        128-bit GOID hash.
    urn
        GOID URN string.
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Relative file path.
    language
        Programming language.
    kind
        Entity kind (module, class, function, method).
    qualname
        Fully qualified name.
    start_line
        Starting line number.
    end_line
        Ending line number.
    created_at
        Creation timestamp.
    """

    goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    start_line: int | None
    end_line: int | None
    created_at: datetime

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class GoidCrosswalkRow:
    """Row data for core.goid_crosswalk table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    goid
        GOID URN string.
    lang
        Programming language.
    module_path
        Module path.
    file_path
        File path.
    start_line
        Starting line number.
    end_line
        Ending line number.
    scip_symbol
        Optional SCIP symbol identifier.
    ast_qualname
        AST qualified name.
    cst_node_id
        Optional CST node identifier.
    chunk_id
        Optional chunk identifier.
    symbol_id
        Optional symbol identifier.
    updated_at
        Update timestamp.
    """

    repo: str
    commit: str
    goid: str
    lang: str
    module_path: str
    file_path: str
    start_line: int | None
    end_line: int | None
    scip_symbol: str | None
    ast_qualname: str | None
    cst_node_id: str | None
    chunk_id: str | None
    symbol_id: str | None
    updated_at: datetime

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class ScipSymbolRow:
    """Row data for core.scip_symbols table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Repository-relative file path.
    symbol
        SCIP symbol identifier.
    documentation
        Optional symbol documentation.
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    rel_path: str
    symbol: str
    documentation: str | None
    created_at: datetime

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class ScipOccurrenceRow:
    """Row data for core.scip_occurrences table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Repository-relative file path.
    symbol
        SCIP symbol identifier.
    start_line
        Starting line number (0-based).
    start_col
        Starting column number (0-based).
    end_line
        Ending line number (0-based).
    end_col
        Ending column number (0-based).
    roles
        SCIP symbol roles bitmask.
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    rel_path: str
    symbol: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    roles: int
    created_at: datetime

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class ScipSymbolInformationRow:
    """Row data for core.scip_symbol_information table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    symbol
        SCIP symbol identifier.
    documentation
        Optional symbol documentation.
    kind
        Symbol kind enum value.
    display_name
        Optional display name for the symbol.
    signature
        Optional signature documentation string.
    enclosing_symbol
        Optional enclosing symbol identifier.
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    symbol: str
    documentation: str | None
    kind: int | None
    display_name: str | None
    signature: str | None
    enclosing_symbol: str | None
    created_at: datetime

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class ScipSymbolRelationshipRow:
    """Row data for core.scip_symbol_relationships table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    symbol
        SCIP symbol identifier.
    related_symbol
        Related SCIP symbol identifier.
    relationship_kind
        Relationship kind label (implementation, reference, type_definition, definition).
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    symbol: str
    related_symbol: str
    relationship_kind: str
    created_at: datetime

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class ScipDiagnosticRow:
    """Row data for core.scip_diagnostics table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Repository-relative file path.
    start_line
        Diagnostic start line (0-based).
    start_col
        Diagnostic start column (0-based).
    end_line
        Diagnostic end line (0-based).
    end_col
        Diagnostic end column (0-based).
    severity
        Diagnostic severity string.
    code
        Optional diagnostic code.
    message
        Diagnostic message.
    source
        Optional diagnostic source.
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    rel_path: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    severity: str
    code: str | None
    message: str
    source: str | None
    created_at: datetime

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class ScipExternalSymbolRow:
    """Row data for core.scip_external_symbols table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    symbol
        SCIP symbol identifier.
    package_manager
        Optional package manager (e.g., pypi).
    package_name
        Optional package name.
    package_version
        Optional package version.
    created_at
        Row creation timestamp.
    """

    repo: str
    commit: str
    symbol: str
    package_manager: str | None
    package_name: str | None
    package_version: str | None
    created_at: datetime

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


@dataclass(frozen=True)
class ScipModuleStateRow:
    """Row data for core.scip_module_state table.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Repository-relative file path.
    content_hash
        Content hash of the module.
    options_hash
        Optional options hash applied during indexing.
    tool_version
        Optional scip-python version string.
    shard_path
        Filesystem path to the module shard.
    updated_at
        Timestamp of the last shard update.
    """

    repo: str
    commit: str
    rel_path: str
    content_hash: str
    options_hash: str | None
    tool_version: str | None
    shard_path: str
    updated_at: datetime

    def to_tuple(self) -> tuple[object, ...]:
        """Serialize to tuple for DuckDB insertion.

        Returns
        -------
        tuple[object, ...]
            Field values in INSERT column order.
        """
        return dataclasses.astuple(self)


__all__ = [
    "CFGBlockRow",
    "CFGEdgeRow",
    "DFGEdgeRow",
    "GoidCrosswalkRow",
    "GoidRow",
    "ImportEdgeRow",
    "ImportModuleRow",
    "ScipDiagnosticRow",
    "ScipExternalSymbolRow",
    "ScipModuleStateRow",
    "ScipOccurrenceRow",
    "ScipSymbolInformationRow",
    "ScipSymbolRelationshipRow",
    "ScipSymbolRow",
    "SymbolUseRow",
]
