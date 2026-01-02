"""Contract lookup handler (Q5)."""

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


def _contract_patterns(name: str) -> dict[str, object]:
    escaped = re.escape(name)
    return {
        "pattern_group_id": f"rg.contract.lookup.{name}",
        "patterns": [
            {
                "pattern": rf"\b{escaped}\b",
                "is_regex": True,
                "priority": 10,
            }
        ],
        "globs": [
            "**/tests/**/*.py",
            "**/test_*.py",
            "**/*_test.py",
            "**/docs/**/*.md",
            "**/README*",
            "**/examples/**/*",
        ],
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
    """Locate tests/docs/examples referencing a symbol.

    Parameters
    ----------
    request:
        Query request containing the symbol name.
    context:
        Search context providing indices and catalogs.

    Returns
    -------
    QueryResponse
        Query response with matching contract references.
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
    pattern_group = _contract_patterns(name)
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

    matches: list[dict[str, object]] = []
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
        matches.append(
            {
                "path": span.path,
                "span": span.to_dict(),
                "snippet": snippet.to_dict(),
            }
        )

    summary = f"Found {len(matches)} contract reference(s) for '{name}'."
    debug = {
        "rg_files": sorted(result.files_to_patterns.keys()),
        "rg_partial": result.partial,
    }
    return QueryResponse(summary=summary, primary=matches, related={}, debug=debug)


__all__ = ["handle"]
