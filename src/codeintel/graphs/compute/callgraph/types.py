"""Data types for call graph computation.

This module defines the core data classes used throughout call graph
construction and resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.graphs.catalog import FunctionSpanIndex


@dataclass(frozen=True)
class CallEdge:
    """Represent a call graph edge.

    Attributes
    ----------
    caller_goid
        GOID of the calling function.
    callee_goid
        GOID of the called function (None if unresolved).
    callee_name
        Name of the called function.
    call_line
        Line number of the call.
    rel_path
        Relative file path where call occurs.
    evidence
        Evidence supporting the edge (local, import, global, scip).
    confidence
        Confidence score (0.0 to 1.0).
    """

    caller_goid: int
    callee_goid: int | None
    callee_name: str
    call_line: int
    rel_path: str
    evidence: str
    confidence: float


@dataclass(frozen=True)
class ResolutionResult:
    """Structured outcome for a single callee resolution attempt.

    Attributes
    ----------
    callee_goid
        Resolved GOID or None if unresolved.
    resolved_via
        How the resolution was achieved.
    confidence
        Confidence score.
    """

    callee_goid: int | None
    resolved_via: str
    confidence: float


@dataclass
class ResolutionContext:
    """Context for call resolution operations.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    local_callees
        Local name to GOID mapping for current file.
    global_callees
        Global name to GOID mapping across repository.
    import_aliases
        Import alias to module mapping.
    scip_candidates
        SCIP-derived candidates by use path.
    def_goids_by_path
        Definition GOIDs by file path.
    """

    repo: str
    commit: str
    local_callees: Mapping[str, int] = field(default_factory=dict)
    global_callees: Mapping[str, int] = field(default_factory=dict)
    import_aliases: Mapping[str, str] = field(default_factory=dict)
    scip_candidates: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    def_goids_by_path: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeResolutionContext:
    """Resolution helpers shared across call graph visitors.

    This context provides all the mappings needed for resolving callees
    during edge collection traversals.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    function_index
        Index for looking up function spans.
    local_callees
        Local name to GOID mapping.
    global_callees
        Global name to GOID mapping.
    import_aliases
        Import alias to module mapping.
    scip_candidates_by_use_path
        SCIP candidates indexed by use path.
    def_goids_by_path
        Definition GOIDs indexed by path.
    """

    repo: str
    commit: str
    function_index: FunctionSpanIndex
    local_callees: dict[str, int]
    global_callees: dict[str, int]
    import_aliases: dict[str, str]
    scip_candidates_by_use_path: dict[str, tuple[str, ...]]
    def_goids_by_path: dict[str, int]


__all__ = [
    "CallEdge",
    "EdgeResolutionContext",
    "ResolutionContext",
    "ResolutionResult",
]
