"""Pure resolution logic for call graph edges."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.ingestion.paths import normalize_rel_path


@dataclass(frozen=True)
class ResolutionResult:
    """Structured outcome for a single callee resolution attempt."""

    callee_goid: int | None
    resolved_via: str
    confidence: float


def resolve_callee(
    callee_name: str,
    attr_chain: list[str],
    local_callees: dict[str, int],
    global_callees: dict[str, int],
    import_aliases: dict[str, str],
) -> ResolutionResult:
    """
    Resolve a callee GOID using local/global maps and import aliases.

    Resolution precedence: local name -> local attr -> import alias -> global name -> global attr.

    Parameters
    ----------
    callee_name : str
        Base callee name extracted from the call expression.
    attr_chain : list[str]
        Attribute chain on the callee (e.g., ["module", "func"]).
    local_callees : dict[str, int]
        Mapping of locally defined callables to GOIDs.
    global_callees : dict[str, int]
        Mapping of repository-global callables to GOIDs.
    import_aliases : dict[str, str]
        Import alias mapping from local alias to fully qualified module path.

    Returns
    -------
    ResolutionResult
        Structured resolution outcome with callee GOID, provenance, and confidence.
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
    candidate_def_paths: tuple[str, ...], def_goids_by_path: dict[str, int]
) -> ResolutionResult:
    """
    Resolve using SCIP definition paths when primary resolution fails.

    Parameters
    ----------
    candidate_def_paths : tuple[str, ...]
        Candidate definition paths produced by SCIP cross-references.
    def_goids_by_path : dict[str, int]
        Mapping from normalized definition paths to GOIDs.

    Returns
    -------
    ResolutionResult
        Resolution outcome using SCIP data or unresolved when none match.
    """
    for def_path in candidate_def_paths:
        goid = def_goids_by_path.get(normalize_rel_path(def_path))
        if goid is not None:
            return ResolutionResult(callee_goid=goid, resolved_via="scip_def_path", confidence=0.55)
    return ResolutionResult(callee_goid=None, resolved_via="unresolved", confidence=0.0)


def build_evidence(
    callee_name: str,
    attr_chain: list[str],
    resolution: ResolutionResult,
    scip_candidates: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """
    Construct evidence payload in a consistent shape.

    Parameters
    ----------
    callee_name : str
        Base callee name extracted from the call expression.
    attr_chain : list[str]
        Attribute chain on the callee; empty when no attributes are present.
    resolution : ResolutionResult
        Resolution outcome detailing GOID and provenance.
    scip_candidates : tuple[str, ...], optional
        SCIP candidate definition paths if available.

    Returns
    -------
    dict[str, object]
        Evidence payload suitable for persistence or debugging.
    """
    evidence: dict[str, object] = {
        "callee_name": callee_name,
        "attr_chain": attr_chain or None,
        "resolved_via": resolution.resolved_via,
    }
    if scip_candidates:
        evidence["scip_candidates"] = list(scip_candidates)
    return evidence
