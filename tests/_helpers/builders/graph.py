"""Row dataclasses and helpers for graph.* schema tables.

Use ``make_symbol_use_edge_row`` + ``insert_symbol_use_edges`` as the canonical
way to seed ``graph.symbol_use_edges`` in tests to ensure schema correctness and
consistent defaults.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import ClassVar, TypedDict, cast

from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders.row_protocol import insert_rows

__all__ = [
    "CFGBlockRow",
    "CFGEdgeRow",
    "CallGraphEdgeRow",
    "CallGraphNodeRow",
    "DFGEdgeRow",
    "ImportGraphEdgeRow",
    "SymbolUseEdgeInput",
    "SymbolUseEdgeRow",
    "insert_symbol_use_edges",
    "make_symbol_use_edge_row",
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

    def to_tuple(self) -> tuple[str, str, str, bool, bool, int | None, int | None]:
        """Return standard tuple for basic insertion.

        Returns
        -------
        tuple[str, str, str, bool, bool, int | None, int | None]
            Row values in column order including optional GOIDs.
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

    def to_basic_tuple(self) -> tuple[str, str, str, bool, bool, int | None, int | None]:
        """Return tuple for basic insertion including optional GOIDs.

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
            self.def_goid_h128,
            self.use_goid_h128,
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


SymbolUseEdgeInput = SymbolUseEdgeRow | Mapping[str, object] | Sequence[object]

_EXPECTED_SEQUENCE_LENGTHS: set[int] = {5, 7}
_EDGE_FIELD_COUNT_FULL = 7


class _SymbolMapping(TypedDict):
    """TypedDict for symbol_use_edge mappings."""

    symbol: object
    def_path: object
    use_path: object
    same_file: object
    same_module: object
    def_goid_h128: object
    use_goid_h128: object


@dataclass(frozen=True)
class SymbolEdgeOptions:
    """Options for constructing symbol use edge rows."""

    same_file: bool | None = None
    same_module: bool | None = None
    def_goid_h128: int | None = None
    use_goid_h128: int | None = None


def _as_optional_bool(value: object) -> bool | None:
    """Convert a value to bool when already boolean, otherwise allow None.

    Returns
    -------
    bool | None
        Boolean value when provided, otherwise None.

    Raises
    ------
    TypeError
        If value is not None or bool.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    message = f"Expected bool or None for same_file/same_module, got {type(value)}"
    raise TypeError(message)


def _normalize_same_flags(
    def_path: str,
    use_path: str,
    *,
    same_file: bool | None,
    same_module: bool | None,
) -> tuple[bool, bool]:
    """Derive same_file and same_module defaults from paths when omitted.

    Returns
    -------
    tuple[bool, bool]
        Normalized same_file and same_module flags.
    """
    def_path_obj = Path(def_path)
    use_path_obj = Path(use_path)
    normalized_same_file = same_file if same_file is not None else def_path_obj == use_path_obj
    if same_module is not None:
        return normalized_same_file, same_module
    return normalized_same_file, def_path_obj.parent == use_path_obj.parent


def _coerce_goids(
    def_goid_raw: object | None,
    use_goid_raw: object | None,
) -> tuple[int | None, int | None]:
    """Coerce raw GOID values to optional ints.

    Returns
    -------
    tuple[int | None, int | None]
        Definition and use GOIDs coerced to ints when numeric, otherwise None.
    """
    def_val = int(def_goid_raw) if isinstance(def_goid_raw, (int, float, Decimal)) else None
    use_val = int(use_goid_raw) if isinstance(use_goid_raw, (int, float, Decimal)) else None
    return def_val, use_val


def make_symbol_use_edge_row(
    symbol: str,
    def_path: str,
    use_path: str,
    *,
    options: SymbolEdgeOptions | None = None,
) -> SymbolUseEdgeRow:
    """Build a SymbolUseEdgeRow with inferred same-file/module defaults.

    Parameters
    ----------
    symbol
        Symbol identifier.
    def_path
        Definition path.
    use_path
        Use path.
    options
        Optional SymbolEdgeOptions bundle controlling flags and GOIDs.

    Returns
    -------
    SymbolUseEdgeRow
        Constructed edge row with normalized same-file/module flags.

    Examples
    --------
    >>> make_symbol_use_edge_row("sym", "a.py", "a.py")
    SymbolUseEdgeRow(symbol='sym', def_path='a.py', use_path='a.py', same_file=True, ...)
    """
    opts = options or SymbolEdgeOptions()
    normalized_same_file, normalized_same_module = _normalize_same_flags(
        def_path,
        use_path,
        same_file=opts.same_file,
        same_module=opts.same_module,
    )
    return SymbolUseEdgeRow(
        symbol=symbol,
        def_path=def_path,
        use_path=use_path,
        same_file=normalized_same_file,
        same_module=normalized_same_module,
        def_goid_h128=opts.def_goid_h128,
        use_goid_h128=opts.use_goid_h128,
    )


def _coerce_symbol_use_edge_row(row: SymbolUseEdgeInput) -> SymbolUseEdgeRow:
    """Normalize supported input shapes into a SymbolUseEdgeRow.

    Returns
    -------
    SymbolUseEdgeRow
        Coerced row with normalized defaults.

    Raises
    ------
    TypeError
        If the row is neither a SymbolUseEdgeRow, mapping, nor sequence.
    ValueError
        If required keys are missing or sequence length is invalid.
    """
    if isinstance(row, SymbolUseEdgeRow):
        return row
    if isinstance(row, Mapping):
        mapping_row = cast("_SymbolMapping", row)
        try:
            symbol = mapping_row["symbol"]
            def_path = mapping_row["def_path"]
            use_path = mapping_row["use_path"]
        except KeyError as exc:
            message = "symbol_use_edge mapping missing required key"
            raise ValueError(message) from exc
        same_file = _as_optional_bool(mapping_row.get("same_file"))
        same_module = _as_optional_bool(mapping_row.get("same_module"))
        def_goid_h128, use_goid_h128 = _coerce_goids(
            mapping_row.get("def_goid_h128"),
            mapping_row.get("use_goid_h128"),
        )
        normalized_same_file, normalized_same_module = _normalize_same_flags(
            str(def_path),
            str(use_path),
            same_file=same_file,
            same_module=same_module,
        )
        return SymbolUseEdgeRow(
            symbol=str(symbol),
            def_path=str(def_path),
            use_path=str(use_path),
            same_file=normalized_same_file,
            same_module=normalized_same_module,
            def_goid_h128=(int(def_goid_h128) if isinstance(def_goid_h128, (int, float)) else None),
            use_goid_h128=(int(use_goid_h128) if isinstance(use_goid_h128, (int, float)) else None),
        )
    if not isinstance(row, Sequence):
        message = f"Unsupported symbol_use_edge row type: {type(row)}"
        raise TypeError(message)
    length = len(row)
    if length not in _EXPECTED_SEQUENCE_LENGTHS:
        message = f"symbol_use_edges rows must have 5 or 7 fields, got {length}: {row}"
        raise ValueError(message)
    symbol, def_path, use_path, same_file, same_module = row[:5]
    def_goid_h128: int | None = None
    use_goid_h128: int | None = None
    if length == _EDGE_FIELD_COUNT_FULL:
        def_goid_h128, use_goid_h128 = _coerce_goids(row[5], row[6])
    normalized_same_file, normalized_same_module = _normalize_same_flags(
        str(def_path),
        str(use_path),
        same_file=_as_optional_bool(same_file),
        same_module=_as_optional_bool(same_module),
    )
    return SymbolUseEdgeRow(
        symbol=str(symbol),
        def_path=str(def_path),
        use_path=str(use_path),
        same_file=normalized_same_file,
        same_module=normalized_same_module,
        def_goid_h128=def_goid_h128,
        use_goid_h128=use_goid_h128,
    )


def insert_symbol_use_edges(
    gateway: StorageGateway,
    rows: Iterable[SymbolUseEdgeInput],
    *,
    coerce_to_full: bool = True,
) -> int:
    """Insert symbol_use_edges rows with schema-aware defaults and validation.

    Canonical helper for tests: prefer this over direct gateway calls to ensure
    NOT NULL fields are filled, same_file/same_module defaults are applied, and
    optional GOIDs are normalized.

    Parameters
    ----------
    gateway
        Storage gateway providing database connection.
    rows
        Iterable of SymbolUseEdgeRow, mapping, or 5/7-field sequence inputs.
    coerce_to_full
        When True (default), insert all seven columns; otherwise insert five.

    Returns
    -------
    int
        Number of rows inserted.
    """
    row_list = list(rows)
    if not row_list:
        return 0

    normalized_rows = [_coerce_symbol_use_edge_row(row) for row in row_list]
    if coerce_to_full:
        gateway.graph.insert_symbol_use_edges([edge.to_full_tuple() for edge in normalized_rows])
        return len(normalized_rows)

    insert_rows(gateway, normalized_rows)
    return len(normalized_rows)
