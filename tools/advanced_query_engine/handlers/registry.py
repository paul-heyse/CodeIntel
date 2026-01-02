"""Handler registry for query types."""

from __future__ import annotations

from collections.abc import Callable

from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import QueryRequest, QueryResponse
from tools.advanced_query_engine.handlers.q1_symbol_resolve import handle as handle_symbol_resolve
from tools.advanced_query_engine.handlers.q2_find_usages import handle as handle_find_usages
from tools.advanced_query_engine.handlers.q3_call_paths import handle as handle_call_paths
from tools.advanced_query_engine.handlers.q4_pattern_scan import handle as handle_pattern_scan
from tools.advanced_query_engine.handlers.q5_contract_lookup import handle as handle_contract_lookup
from tools.advanced_query_engine.handlers.q6_wiring_map import handle as handle_wiring_map
from tools.advanced_query_engine.handlers.q7_precedent_search import (
    handle as handle_precedent_search,
)
from tools.advanced_query_engine.handlers.q8_impact_slice import handle as handle_impact_slice

HandlerFn = Callable[[QueryRequest, SearchContext], QueryResponse]

HANDLERS: dict[str, HandlerFn] = {
    "symbol.resolve": handle_symbol_resolve,
    "refs.find": handle_find_usages,
    "callgraph.slice": handle_call_paths,
    "pattern.scan": handle_pattern_scan,
    "contract.lookup": handle_contract_lookup,
    "wiring.map": handle_wiring_map,
    "precedent.search": handle_precedent_search,
    "impact.slice": handle_impact_slice,
}

__all__ = ["HANDLERS", "HandlerFn"]
