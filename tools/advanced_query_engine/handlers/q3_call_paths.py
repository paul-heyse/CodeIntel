"""Call hierarchy handler (Q3)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from tools.advanced_query_engine.backends.rpygrep_backend import RpygrepQuery, run_pattern_group
from tools.advanced_query_engine.backends.treesitter_backend import (
    TreeSitterCapture,
    TreeSitterQuery,
    TreeSitterRequest,
    run_query_packs,
)
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import QueryBudget, QueryRequest, QueryResponse, Span
from tools.advanced_query_engine.handlers.common import load_rpygrep_preset
from tools.advanced_query_engine.util.semantic_helpers import (
    callee_label,
    map_args_to_params,
    parse_call_args,
    parse_signature,
)
from tools.advanced_query_engine.util.snippets import SnippetRequest, build_snippet


def _call_candidates(name: str) -> dict[str, object]:
    escaped = re.escape(name)
    return {
        "pattern_group_id": f"rg.symbol.calls.{name}",
        "patterns": [
            {
                "pattern": rf"\b{escaped}\b",
                "is_regex": True,
                "priority": 10,
            }
        ],
        "globs": ["**/*.py"],
    }


def _load_call_query(context: SearchContext) -> str:
    try:
        return context.query_catalog.tree_sitter_pack("ts.python.calls")
    except ValueError:
        return ""


@dataclass(frozen=True)
class _CallSite:
    path: str
    call_span: Span
    callee_text: str
    callee_name: str
    args_text: str | None


def _deadline(budget: QueryBudget) -> float | None:
    if budget.max_seconds is None:
        return None
    return time.monotonic() + float(budget.max_seconds)


def _budget_exhausted(budget: QueryBudget, deadline: float | None, count: int) -> bool:
    return (budget.max_matches > 0 and count >= budget.max_matches) or (
        deadline is not None and time.monotonic() >= deadline
    )


def _within(outer: Span, inner: Span) -> bool:
    return outer.start_byte <= inner.start_byte and inner.end_byte <= outer.end_byte


def _select_closest(anchor: Span, candidates: list[Span], *, mode: str) -> Span | None:
    valid = [span for span in candidates if _within(anchor, span)]
    if not valid:
        return None
    if mode == "start":
        return min(valid, key=lambda span: abs(span.start_byte - anchor.start_byte))
    return min(valid, key=lambda span: abs(span.end_byte - anchor.end_byte))


def _call_captures(
    context: SearchContext,
    rel_path: str,
    query_text: str,
    budget: QueryBudget,
) -> list[TreeSitterCapture]:
    if not query_text:
        return []
    source_bytes = context.cache.read_bytes(rel_path)
    parsed = context.tree_sitter_parse(rel_path, "python")
    match_limit = budget.max_matches if budget.max_matches else 2000
    result = run_query_packs(
        TreeSitterRequest(
            language="python",
            source_bytes=source_bytes,
            path=rel_path,
            queries=[TreeSitterQuery(pack_id="ts.python.calls", query_text=query_text)],
            match_limit=match_limit,
            preview_limit=200,
            parsed=parsed,
        )
    )
    return result.captures


def _call_sites(captures: list[TreeSitterCapture]) -> list[_CallSite]:
    call_nodes = [cap for cap in captures if cap.capture_name == "call.node"]
    callee_caps = [cap for cap in captures if cap.capture_name == "call.callee"]
    args_caps = [cap for cap in captures if cap.capture_name == "call.args"]
    callee_spans = [cap.span for cap in callee_caps]
    args_spans = [cap.span for cap in args_caps]
    text_by_span = {
        cap.span: cap.text_preview
        for cap in callee_caps + args_caps
        if cap.text_preview is not None
    }
    call_sites: list[_CallSite] = []
    for node in call_nodes:
        callee_span = _select_closest(node.span, callee_spans, mode="start")
        if callee_span is None:
            continue
        args_span = _select_closest(node.span, args_spans, mode="end")
        callee_text = text_by_span.get(callee_span, "")
        if not callee_text:
            continue
        args_text = text_by_span.get(args_span) if args_span is not None else None
        call_sites.append(
            _CallSite(
                path=node.span.path,
                call_span=node.span,
                callee_text=callee_text,
                callee_name=callee_label(callee_text),
                args_text=args_text,
            )
        )
    return call_sites


def _callsite_args(callsite: _CallSite) -> dict[str, object] | None:
    args_text = callsite.args_text or "()"
    call_expr = f"{callsite.callee_text}{args_text}"
    parsed = parse_call_args(call_expr)
    if parsed is None:
        return None
    return {
        "positional": parsed.positional,
        "keywords": parsed.keywords,
        "has_vararg": parsed.has_vararg,
        "has_kwarg": parsed.has_kwarg,
    }


def _arg_mapping(callsite: _CallSite, signature: str | None) -> dict[str, str] | None:
    if signature is None:
        return None
    args_text = callsite.args_text or "()"
    call_expr = f"{callsite.callee_text}{args_text}"
    parsed_args = parse_call_args(call_expr)
    signature_info = parse_signature(signature)
    if parsed_args is None or signature_info is None:
        return None
    return map_args_to_params(signature_info, parsed_args)


def _edge_payload(
    *,
    callsite: _CallSite,
    context: SearchContext,
    caller: dict[str, object] | None,
    callee: dict[str, object] | None,
    arg_map: dict[str, str] | None,
) -> dict[str, object]:
    snippet = build_snippet(
        SnippetRequest(
            source=context.cache.read_bytes(callsite.path),
            span=callsite.call_span,
            config=context.snippet_config,
            line_index=context.line_index(callsite.path),
        )
    )
    payload: dict[str, object] = {
        "caller": caller,
        "callee": callee,
        "call_span": callsite.call_span.to_dict(),
        "callsite": snippet.to_dict(),
        "arguments": _callsite_args(callsite),
    }
    if arg_map:
        payload["arg_map"] = arg_map
    return payload


def _caller_payload(callsite: _CallSite, def_index: object | None) -> dict[str, object] | None:
    if def_index is None:
        return None
    enclosing = def_index.enclosing_def(callsite.call_span.start_byte)
    if enclosing is None:
        return None
    return {
        "name": enclosing.name,
        "qname": enclosing.qname,
        "kind": enclosing.kind,
    }


def _callee_payload(
    callsite: _CallSite, def_index: object | None
) -> tuple[dict[str, object], str | None]:
    payload = {"name": callsite.callee_name, "expr": callsite.callee_text}
    signature = None
    if def_index is not None and callsite.callee_name:
        defs = def_index.by_name(callsite.callee_name)
        if defs:
            def_rec = defs[0]
            signature = def_rec.signature
            payload["resolved"] = {
                "name": def_rec.name,
                "qname": def_rec.qname,
                "kind": def_rec.kind,
                "signature": def_rec.signature,
            }
    return payload, signature


def _incoming_edges(
    name: str,
    call_sites: list[_CallSite],
    def_index: object | None,
    context: SearchContext,
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for callsite in call_sites:
        if callsite.callee_name != name:
            continue
        caller = _caller_payload(callsite, def_index)
        callee, signature = _callee_payload(callsite, def_index)
        arg_map = _arg_mapping(callsite, signature)
        edges.append(
            _edge_payload(
                callsite=callsite,
                context=context,
                caller=caller,
                callee=callee,
                arg_map=arg_map,
            )
        )
    return edges


def _outgoing_edges(
    name: str,
    call_sites: list[_CallSite],
    def_index: object | None,
    context: SearchContext,
) -> list[dict[str, object]]:
    if def_index is None:
        return []
    edges: list[dict[str, object]] = []
    for def_rec in def_index.by_name(name):
        for callsite in call_sites:
            if not _within(def_rec.span, callsite.call_span):
                continue
            caller = {"name": def_rec.name, "qname": def_rec.qname, "kind": def_rec.kind}
            callee, signature = _callee_payload(callsite, def_index)
            arg_map = _arg_mapping(callsite, signature)
            edges.append(
                _edge_payload(
                    callsite=callsite,
                    context=context,
                    caller=caller,
                    callee=callee,
                    arg_map=arg_map,
                )
            )
    return edges


def _edges_for_file(
    name: str,
    context: SearchContext,
    rel_path: str,
    query_text: str,
    budget: QueryBudget,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    captures = _call_captures(context, rel_path, query_text, budget)
    if not captures:
        return [], []
    call_sites = _call_sites(captures)
    try:
        def_index = context.def_index(rel_path)
    except (FileNotFoundError, ValueError):
        def_index = None
    return (
        _incoming_edges(name, call_sites, def_index, context),
        _outgoing_edges(name, call_sites, def_index, context),
    )


def handle(request: QueryRequest, context: SearchContext) -> QueryResponse:
    """Trace call edges for a symbol name.

    Returns
    -------
    QueryResponse
        Call hierarchy response.
    """
    name = request.text.strip()
    if not name:
        return QueryResponse(
            summary="Empty symbol name; no results.",
            primary=[],
            related={},
            debug={"reason": "empty_symbol"},
        )

    budget = request.budget or context.default_budget
    if not isinstance(budget, QueryBudget):
        budget = QueryBudget()
    deadline = _deadline(budget)

    preset = load_rpygrep_preset(context.query_catalog, "rg.default_interactive")
    candidate_result = run_pattern_group(
        RpygrepQuery(
            repo_root=context.repo_root,
            preset=preset,
            pattern_group=_call_candidates(name),
            budget=budget,
            scope_paths=request.scope_paths,
            cache=context.cache,
        )
    )

    query_text = _load_call_query(context)
    if not query_text:
        summary = f"No tree-sitter call pack available for '{name}'."
        return QueryResponse(summary=summary, primary=[], related={}, debug={"reason": "no_calls"})

    call_edges_in: list[dict[str, object]] = []
    call_edges_out: list[dict[str, object]] = []
    budget_exhausted = False

    for rel_path in sorted(candidate_result.files_to_patterns.keys()):
        if _budget_exhausted(budget, deadline, len(call_edges_in) + len(call_edges_out)):
            budget_exhausted = True
            break
        if not rel_path.endswith(".py"):
            continue
        incoming, outgoing = _edges_for_file(name, context, rel_path, query_text, budget)
        call_edges_in.extend(incoming)
        call_edges_out.extend(outgoing)

    summary = (
        f"Found {len(call_edges_in)} incoming and {len(call_edges_out)} outgoing calls "
        f"for '{name}'."
    )
    related = {"calls_in": call_edges_in, "calls_out": call_edges_out}
    debug = {
        "rg_files": sorted(candidate_result.files_to_patterns.keys()),
        "rg_partial": candidate_result.partial,
        "budget_exhausted": budget_exhausted,
    }
    return QueryResponse(summary=summary, primary=[], related=related, debug=debug)


__all__ = ["handle"]
