"""Row dataclasses for graph.* schema tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    "CFGBlockRow",
    "CFGEdgeRow",
    "CallGraphEdgeRow",
    "CallGraphNodeRow",
    "DFGEdgeRow",
    "ImportGraphEdgeRow",
    "SymbolUseEdgeRow",
]


@dataclass(frozen=True)
class CallGraphNodeRow:
    """Row for graph.call_graph_nodes."""

    __table__: ClassVar[str] = "graph.call_graph_nodes"
    __columns__: ClassVar[tuple[str, ...]] = (
        "goid_h128",
        "language",
        "kind",
        "arity",
        "is_public",
        "rel_path",
    )

    goid_h128: int
    language: str
    kind: str
    arity: int
    is_public: bool
    rel_path: str

    def to_tuple(self) -> tuple[int, str, str, int, bool, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.goid_h128,
            self.language,
            self.kind,
            self.arity,
            self.is_public,
            self.rel_path,
        )


@dataclass(frozen=True)
class CallGraphEdgeRow:
    """Row for graph.call_graph_edges."""

    __table__: ClassVar[str] = "graph.call_graph_edges"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "caller_goid_h128",
        "callee_goid_h128",
        "callsite_path",
        "callsite_line",
        "callsite_col",
        "language",
        "kind",
        "resolved_via",
        "confidence",
        "evidence_json",
    )

    repo: str
    commit: str
    caller_goid_h128: int
    callee_goid_h128: int | None
    callsite_path: str
    callsite_line: int
    callsite_col: int
    language: str
    kind: str
    resolved_via: str
    confidence: float
    evidence: dict[str, object] | str | None = None

    def to_tuple(
        self,
    ) -> tuple[str, str, int, int | None, str, int, int, str, str, str, float, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        evidence_json: str
        if self.evidence is None:
            evidence_json = "{}"
        elif isinstance(self.evidence, str):
            evidence_json = self.evidence
        else:
            evidence_json = json.dumps(self.evidence)
        return (
            self.repo,
            self.commit,
            self.caller_goid_h128,
            self.callee_goid_h128,
            self.callsite_path,
            self.callsite_line,
            self.callsite_col,
            self.language,
            self.kind,
            self.resolved_via,
            self.confidence,
            evidence_json,
        )


@dataclass(frozen=True)
class ImportGraphEdgeRow:
    """Row for graph.import_graph_edges."""

    __table__: ClassVar[str] = "graph.import_graph_edges"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "src_module",
        "dst_module",
        "src_fan_out",
        "dst_fan_in",
        "cycle_group",
    )

    repo: str
    commit: str
    src_module: str
    dst_module: str
    src_fan_out: int
    dst_fan_in: int
    cycle_group: int
    module_layer: int | None = None

    def to_tuple(self) -> tuple[str, str, str, str, int, int, int]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.repo,
            self.commit,
            self.src_module,
            self.dst_module,
            self.src_fan_out,
            self.dst_fan_in,
            self.cycle_group,
        )


@dataclass(frozen=True)
class SymbolUseEdgeRow:
    """Row for graph.symbol_use_edges with GOID detail."""

    __table__: ClassVar[str] = "graph.symbol_use_edges"
    __columns__: ClassVar[tuple[str, ...]] = (
        "symbol",
        "def_path",
        "use_path",
        "same_file",
        "same_module",
        "def_goid_h128",
        "use_goid_h128",
    )

    symbol: str
    def_path: str
    use_path: str
    same_file: bool
    same_module: bool
    def_goid_h128: int | None = None
    use_goid_h128: int | None = None

    def to_tuple(self) -> tuple[str, str, str, bool, bool]:
        """Return standard tuple for basic insertion.

        Returns
        -------
        tuple[str, str, str, bool, bool]
            Row values in column order.
        """
        return (
            self.symbol,
            self.def_path,
            self.use_path,
            self.same_file,
            self.same_module,
        )

    def to_full_tuple(
        self,
    ) -> tuple[str, str, str, bool, bool, int | None, int | None]:
        """Return tuple with all columns, including optional GOID fields.

        Returns
        -------
        tuple
            Values including GOID fields in column order.
        """
        return (
            self.symbol,
            self.def_path,
            self.use_path,
            self.same_file,
            self.same_module,
            self.def_goid_h128,
            self.use_goid_h128,
        )

    def to_basic_tuple(self) -> tuple[str, str, str, bool, bool]:
        """Return tuple for basic insertion.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.symbol,
            self.def_path,
            self.use_path,
            self.same_file,
            self.same_module,
        )

    def to_detailed_tuple(
        self,
    ) -> tuple[str, str, str, bool, bool, int | None, int | None]:
        """Return tuple with GOID details.

        Returns
        -------
        tuple
            Values including optional GOID fields.
        """
        return (
            self.symbol,
            self.def_path,
            self.use_path,
            self.same_file,
            self.same_module,
            self.def_goid_h128,
            self.use_goid_h128,
        )


@dataclass(frozen=True)
class CFGBlockRow:
    """Row for graph.cfg_blocks."""

    __table__: ClassVar[str] = "graph.cfg_blocks"
    __columns__: ClassVar[tuple[str, ...]] = (
        "function_goid_h128",
        "block_idx",
        "block_id",
        "label",
        "file_path",
        "start_line",
        "end_line",
        "kind",
        "stmts_json",
        "in_degree",
        "out_degree",
    )

    function_goid_h128: int
    block_idx: int
    block_id: str
    label: str
    file_path: str
    start_line: int
    end_line: int
    kind: str
    stmts_json: str
    in_degree: int
    out_degree: int

    def to_tuple(
        self,
    ) -> tuple[int, int, str, str, str, int, int, str, str, int, int]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.function_goid_h128,
            self.block_idx,
            self.block_id,
            self.label,
            self.file_path,
            self.start_line,
            self.end_line,
            self.kind,
            self.stmts_json,
            self.in_degree,
            self.out_degree,
        )


@dataclass(frozen=True)
class CFGEdgeRow:
    """Row for graph.cfg_edges."""

    __table__: ClassVar[str] = "graph.cfg_edges"
    __columns__: ClassVar[tuple[str, ...]] = (
        "function_goid_h128",
        "src_block_id",
        "dst_block_id",
        "edge_kind",
    )

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    edge_kind: str | None

    def to_tuple(self) -> tuple[int, str, str, str | None]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.function_goid_h128,
            self.src_block_id,
            self.dst_block_id,
            self.edge_kind,
        )


@dataclass(frozen=True)
class DFGEdgeRow:
    """Row for graph.dfg_edges."""

    __table__: ClassVar[str] = "graph.dfg_edges"
    __columns__: ClassVar[tuple[str, ...]] = (
        "function_goid_h128",
        "src_block_id",
        "dst_block_id",
        "src_var",
        "dst_var",
        "edge_kind",
        "via_phi",
        "use_kind",
    )

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    src_var: str | None
    dst_var: str | None
    edge_kind: str | None
    via_phi: bool
    use_kind: str | None

    def to_tuple(
        self,
    ) -> tuple[int, str, str, str | None, str | None, str | None, bool, str | None]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.function_goid_h128,
            self.src_block_id,
            self.dst_block_id,
            self.src_var,
            self.dst_var,
            self.edge_kind,
            self.via_phi,
            self.use_kind,
        )
