"""Impact slice handler (Q8)."""

from __future__ import annotations

from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import QueryRequest, QueryResponse
from tools.advanced_query_engine.handlers.q2_find_usages import handle as handle_find_usages
from tools.advanced_query_engine.handlers.q3_call_paths import handle as handle_call_paths
from tools.advanced_query_engine.handlers.q6_wiring_map import handle as handle_wiring_map


def handle(request: QueryRequest, context: SearchContext) -> QueryResponse:
    """Compute a bounded impact slice for a symbol name.

    Parameters
    ----------
    request:
        Query request containing the symbol name.
    context:
        Search context providing indices and catalogs.

    Returns
    -------
    QueryResponse
        Query response with aggregated impact data.
    """
    refs_response = handle_find_usages(request, context)
    calls_response = handle_call_paths(request, context)

    options = request.options or {}
    include_wiring = bool(options.get("include_wiring", False))
    wiring_response = None
    if include_wiring:
        wiring_response = handle_wiring_map(request, context)

    related = {
        "references": refs_response.primary,
        "calls_in": calls_response.related.get("calls_in", []),
        "calls_out": calls_response.related.get("calls_out", []),
    }
    if wiring_response is not None:
        related["wiring"] = wiring_response.primary

    summary = (
        f"Impact slice: {len(refs_response.primary)} refs, "
        f"{len(calls_response.related.get('calls_in', []))} incoming calls."
    )
    debug = {
        "refs_debug": refs_response.debug,
        "calls_debug": calls_response.debug,
    }
    if wiring_response is not None:
        debug["wiring_debug"] = wiring_response.debug

    return QueryResponse(summary=summary, primary=[], related=related, debug=debug)


__all__ = ["handle"]
