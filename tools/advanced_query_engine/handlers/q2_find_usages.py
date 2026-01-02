"""Reference discovery handler (Q2)."""

from __future__ import annotations

import re

from tools.advanced_query_engine.backends.rpygrep_backend import (
    RpygrepMatch,
    RpygrepQuery,
    run_pattern_group,
)
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import QueryBudget, QueryRequest, QueryResponse, Span
from tools.advanced_query_engine.handlers.common import load_rpygrep_preset
from tools.advanced_query_engine.util.snippets import SnippetRequest, build_snippet


def _usage_patterns(name: str) -> dict[str, object]:
    escaped = re.escape(name)
    return {
        "pattern_group_id": f"rg.symbol.refs.{name}",
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


def _span_from_match(match: RpygrepMatch, context: SearchContext) -> Span | None:
    if match.span is not None:
        return match.span
    try:
        index = context.line_index(match.path)
    except FileNotFoundError:
        return None
    line_start = index.line_start_byte(match.line_number)
    return Span(
        path=match.path,
        start_byte=line_start + match.submatch_start,
        end_byte=line_start + match.submatch_end,
        **index.span_to_range(line_start + match.submatch_start, line_start + match.submatch_end),
    )


def handle(request: QueryRequest, context: SearchContext) -> QueryResponse:
    """Enumerate lexical usages of a symbol name.

    Parameters
    ----------
    request:
        Query request containing the symbol name.
    context:
        Search context providing indices and catalogs.

    Returns
    -------
    QueryResponse
        Query response with matching usage records.
    """
    name = request.text.strip()
    if not name:
        return QueryResponse(
            summary="Empty symbol name; no results.",
            primary=[],
            related={},
            debug={"reason": "empty_symbol"},
        )

    preset = load_rpygrep_preset(context.query_catalog, "rg.default_interactive")
    pattern_group = _usage_patterns(name)
    budget = request.budget or context.default_budget
    if not isinstance(budget, QueryBudget):
        budget = QueryBudget()

    result = run_pattern_group(
        RpygrepQuery(
            repo_root=context.repo_root,
            preset=preset,
            pattern_group=pattern_group,
            budget=budget,
            scope_paths=request.scope_paths,
            cache=context.cache,
        )
    )

    records: list[dict[str, object]] = []
    for match in result.matches[: budget.max_matches]:
        span = _span_from_match(match, context)
        if span is None:
            continue
        snippet = build_snippet(
            SnippetRequest(
                source=context.cache.read_bytes(span.path),
                span=span,
                config=context.snippet_config,
                line_index=context.line_index(span.path),
            )
        )
        enclosing = None
        try:
            def_index = context.def_index(span.path)
            enclosing = def_index.enclosing_def(span.start_byte)
        except (FileNotFoundError, ValueError):
            enclosing = None

        record = {
            "path": span.path,
            "span": span.to_dict(),
            "snippet": snippet.to_dict(),
        }
        if enclosing is not None:
            record["enclosing"] = {
                "name": enclosing.name,
                "qname": enclosing.qname,
                "kind": enclosing.kind,
            }
        records.append(record)

    summary = f"Found {len(records)} reference(s) for '{name}'."
    debug = {
        "rg_partial": result.partial,
        "rg_files": sorted(result.files_to_patterns.keys()),
    }
    return QueryResponse(summary=summary, primary=records, related={}, debug=debug)


__all__ = ["handle"]
