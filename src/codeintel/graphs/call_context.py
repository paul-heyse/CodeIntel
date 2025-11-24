"""Shared context structures for call graph construction."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.graphs.function_index import FunctionSpanIndex


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
