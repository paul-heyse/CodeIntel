"""Graph table row models and serializers.

This module provides row types for graph DuckDB tables:
- CallGraphNodeRow for graph.call_graph_nodes (TypedDict - not duplicated in compute)
- CallGraphEdgeRow for graph.call_graph_edges (TypedDict - not duplicated in compute)
- Other row types are re-exported from codeintel.graphs.data_models.rows (canonical source)

The canonical dataclass definitions for CFG, DFG, Import, and Symbol rows
live in codeintel.graphs.data_models.rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

# Re-export canonical row types from data_models (single source of truth)
from codeintel.core.data_models.rows import (
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    ImportEdgeRow,
    ImportModuleRow,
    SymbolUseRow,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


# Simple alias to align with serializer naming used in tests and contracts.
def symbol_use_to_tuple(row: SymbolUseRow) -> tuple[object, ...]:
    """Serialize a SymbolUseRow into the INSERT column order.

    Returns
    -------
    tuple[object, ...]
        Values aligned with symbol_use INSERT order.
    """
    return SymbolUseRow.to_tuple(row)


# Call graph types remain TypedDicts as they are not duplicated in compute modules


class CallGraphNodeRow(TypedDict):
    """Row shape for graph.call_graph_nodes inserts.

    Parameters
    ----------
    goid_h128
        128-bit hash of the function GOID.
    language
        Programming language.
    kind
        Function kind.
    arity
        Number of parameters.
    is_public
        Whether the function is public.
    rel_path
        Relative file path.
    """

    goid_h128: int
    language: str
    kind: str
    arity: int
    is_public: bool
    rel_path: str


def call_graph_node_to_tuple(row: CallGraphNodeRow) -> tuple[object, ...]:
    """Serialize a CallGraphNodeRow into the INSERT column order.

    Parameters
    ----------
    row
        The call graph node row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with call_graph_nodes INSERT order.
    """
    return (
        row["goid_h128"],
        row["language"],
        row["kind"],
        row["arity"],
        row["is_public"],
        row["rel_path"],
    )


def dict_to_call_graph_node(row: Mapping[str, object]) -> CallGraphNodeRow:
    """Cast a dictionary to CallGraphNodeRow type.

    This is a type-safe adapter for converting validated dictionary data
    back to the typed row format expected by serialization functions.

    Parameters
    ----------
    row
        Dictionary with call graph node fields.

    Returns
    -------
    CallGraphNodeRow
        Typed row suitable for serialization.
    """
    return cast("CallGraphNodeRow", dict(row))


class CallGraphEdgeRow(TypedDict):
    """Row shape for graph.call_graph_edges inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    caller_goid_h128
        Caller function GOID hash.
    callee_goid_h128
        Callee function GOID hash.
    callsite_path
        Path to callsite file.
    callsite_line
        Callsite line number.
    callsite_col
        Callsite column number.
    language
        Programming language.
    kind
        Call kind.
    resolved_via
        Resolution method used.
    confidence
        Resolution confidence score.
    evidence_json
        Evidence for the call (JSON).
    """

    repo: str
    commit: str
    caller_goid_h128: int
    callee_goid_h128: int | None
    callsite_path: str
    callsite_line: int
    callsite_col: int
    language: str
    kind: str
    resolved_via: str | None
    confidence: float | None
    evidence_json: object


def call_graph_edge_to_tuple(row: CallGraphEdgeRow) -> tuple[object, ...]:
    """Serialize a CallGraphEdgeRow into the INSERT column order.

    Parameters
    ----------
    row
        The call graph edge row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values aligned with call_graph_edges INSERT order.
    """
    return (
        row["repo"],
        row["commit"],
        row["caller_goid_h128"],
        row["callee_goid_h128"],
        row["callsite_path"],
        row["callsite_line"],
        row["callsite_col"],
        row["language"],
        row["kind"],
        row["resolved_via"],
        row["confidence"],
        row["evidence_json"],
    )


def dict_to_call_graph_edge(row: Mapping[str, object]) -> CallGraphEdgeRow:
    """Cast a dictionary to CallGraphEdgeRow type.

    This is a type-safe adapter for converting validated dictionary data
    back to the typed row format expected by serialization functions.

    Parameters
    ----------
    row
        Dictionary with call graph edge fields.

    Returns
    -------
    CallGraphEdgeRow
        Typed row suitable for serialization.
    """
    return cast("CallGraphEdgeRow", dict(row))


__all__ = [
    "CFGBlockRow",
    "CFGEdgeRow",
    "CallGraphEdgeRow",
    "CallGraphNodeRow",
    "DFGEdgeRow",
    "ImportEdgeRow",
    "ImportModuleRow",
    "SymbolUseRow",
    "call_graph_edge_to_tuple",
    "call_graph_node_to_tuple",
    "dict_to_call_graph_edge",
    "dict_to_call_graph_node",
]
