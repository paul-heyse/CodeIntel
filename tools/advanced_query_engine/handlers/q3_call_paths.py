"""Call hierarchy handler (Q3)."""

from __future__ import annotations

import re

from tools.advanced_query_engine.backends.rpygrep_backend import RpygrepQuery, run_pattern_group
from tools.advanced_query_engine.backends.treesitter_backend import (
    TreeSitterQuery,
    TreeSitterRequest,
    run_query_packs,
)
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import QueryBudget, QueryRequest, QueryResponse
from tools.advanced_query_engine.handlers.common import load_rpygrep_preset


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
        "exclude_globs": ["**/.venv/**", "**/venv/**", "**/site-packages/**"],
    }


def _callee_name(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return ""
    return expr.split(".")[-1]


def _load_call_query(context: SearchContext) -> str:
    try:
        return context.query_catalog.tree_sitter_pack("ts.python.calls")
    except ValueError:
        return ""


def _call_captures(
    context: SearchContext,
    rel_path: str,
    query_text: str,
    budget: QueryBudget,
) -> list[object]:
    if not query_text:
        return []
    source_bytes = context.cache.read_bytes(rel_path)
    parsed = context.tree_sitter_parse(rel_path, "python")
    result = run_query_packs(
        TreeSitterRequest(
            language="python",
            source_bytes=source_bytes,
            path=rel_path,
            queries=[TreeSitterQuery(pack_id="ts.python.calls", query_text=query_text)],
            match_limit=budget.max_matches,
            preview_limit=200,
            parsed=parsed,
        )
    )
    return [cap for cap in result.captures if cap.capture_name == "call.callee"]


def _incoming_edges(
    name: str, captures: list[object], def_index: object | None
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for cap in captures:
        callee = _callee_name(cap.text_preview or "")
        if callee != name:
            continue
        caller = None
        if def_index is not None:
            caller = def_index.enclosing_def(cap.span.start_byte)
        edges.append(
            {
                "caller": None
                if caller is None
                else {"name": caller.name, "qname": caller.qname, "kind": caller.kind},
                "callee": name,
                "call_span": cap.span.to_dict(),
            }
        )
    return edges


def _outgoing_edges(
    name: str, captures: list[object], def_index: object | None
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    if def_index is None:
        return edges
    for def_rec in def_index.by_name(name):
        for cap in captures:
            if not (def_rec.span.start_byte <= cap.span.start_byte < def_rec.span.end_byte):
                continue
            callee = _callee_name(cap.text_preview or "")
            edges.append(
                {
                    "caller": {"name": def_rec.name, "qname": def_rec.qname, "kind": def_rec.kind},
                    "callee": callee,
                    "call_span": cap.span.to_dict(),
                }
            )
    return edges


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
    call_edges_in: list[dict[str, object]] = []
    call_edges_out: list[dict[str, object]] = []

    for rel_path in sorted(candidate_result.files_to_patterns.keys()):
        if not rel_path.endswith(".py"):
            continue
        captures = _call_captures(context, rel_path, query_text, budget)
        if not captures:
            continue
        try:
            def_index = context.def_index(rel_path)
        except (FileNotFoundError, ValueError):
            def_index = None
        call_edges_in.extend(_incoming_edges(name, captures, def_index))
        call_edges_out.extend(_outgoing_edges(name, captures, def_index))

    summary = f"Found {len(call_edges_in)} incoming and {len(call_edges_out)} outgoing calls for '{name}'."
    related = {
        "calls_in": call_edges_in,
        "calls_out": call_edges_out,
    }
    debug = {
        "rg_files": sorted(candidate_result.files_to_patterns.keys()),
        "rg_partial": candidate_result.partial,
    }
    return QueryResponse(summary=summary, primary=[], related=related, debug=debug)


__all__ = ["handle"]
