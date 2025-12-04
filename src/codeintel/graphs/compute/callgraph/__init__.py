"""Pure call graph computation functions.

This package provides stateless functions for collecting and resolving
call graph edges without any database or file I/O.

Submodules
----------
types
    Data classes for call graph edges and resolution contexts.
resolution
    Functions for resolving callees to GOIDs.
collection
    CST and AST-based edge collection visitors.

Architecture Notes
------------------
This package contains pure computation functions. For persistence, use
``adapters.callgraph_persistence``.
"""

from codeintel.graphs.compute.callgraph.collection import (
    collect_call_sites,
    collect_edges_ast,
    collect_edges_cst,
    collect_edges_for_function,
    dedupe_edges,
    extract_callee_ast,
    extract_callee_cst,
)
from codeintel.graphs.compute.callgraph.resolution import (
    build_callee_map,
    build_evidence,
    collect_aliases,
    collect_import_edges,
    handle_import,
    handle_import_from,
    resolve_callee,
    resolve_via_scip,
)
from codeintel.graphs.compute.callgraph.types import (
    CallEdge,
    EdgeResolutionContext,
    ResolutionContext,
    ResolutionResult,
)

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
