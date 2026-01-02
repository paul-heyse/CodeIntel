"""Symbol resolution handler (Q1)."""

from __future__ import annotations

import re

from tools.advanced_query_engine.backends.rpygrep_backend import RpygrepQuery, run_pattern_group
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import (
    QueryBudget,
    QueryRequest,
    QueryResponse,
    Span,
    SymbolId,
    SymbolRecord,
)
from tools.advanced_query_engine.handlers.common import load_rpygrep_preset
from tools.advanced_query_engine.util.hashing import stable_hex_digest


def _definition_patterns(name: str) -> dict[str, object]:
    escaped = re.escape(name)
    return {
        "pattern_group_id": f"rg.symbol.resolve.{name}",
        "patterns": [
            {
                "pattern": rf"\\bdef\\s+{escaped}\\b",
                "is_regex": True,
                "priority": 10,
            },
            {
                "pattern": rf"\\bclass\\s+{escaped}\\b",
                "is_regex": True,
                "priority": 9,
            },
            {
                "pattern": rf"\\b{escaped}\\s*=",
                "is_regex": True,
                "priority": 5,
            },
        ],
        "globs": ["**/*.py"],
        "exclude_globs": ["**/.venv/**", "**/venv/**", "**/site-packages/**"],
    }


def _span_with_lines(span: Span, context: SearchContext) -> Span:
    index = context.line_index(span.path)
    return Span(
        path=span.path,
        start_byte=span.start_byte,
        end_byte=span.end_byte,
        **index.span_to_range(span.start_byte, span.end_byte),
    )


def handle(request: QueryRequest, context: SearchContext) -> QueryResponse:
    """Resolve symbol definitions by name.

    Parameters
    ----------
    request:
        Query request containing the symbol name.
    context:
        Search context providing indices and catalogs.

    Returns
    -------
    QueryResponse
        Query response with resolved symbol definitions.
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
    pattern_group = _definition_patterns(name)
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

    symbols: list[SymbolRecord] = []
    for rel_path in sorted(result.files_to_patterns.keys()):
        if not rel_path.endswith(".py"):
            continue
        try:
            def_index = context.def_index(rel_path)
        except (FileNotFoundError, ValueError):
            continue
        for record in def_index.by_name(name):
            if budget.max_matches and len(symbols) >= budget.max_matches:
                break
            span = _span_with_lines(record.span, context)
            stable_id = stable_hex_digest(
                [record.path, str(record.span.start_byte), str(record.span.end_byte), record.kind]
            )
            symbol_id = SymbolId(kind=record.kind, stable=stable_id, qname=record.qname)
            symbols.append(
                SymbolRecord(
                    symbol_id=symbol_id,
                    name=record.name,
                    kind=record.kind,
                    def_span=span,
                    signature=record.signature,
                    docstring=record.docstring,
                )
            )
        if budget.max_matches and len(symbols) >= budget.max_matches:
            break

    symbols.sort(key=lambda item: (item.def_span.path, item.def_span.start_byte, item.name))
    primary = [symbol.to_dict() for symbol in symbols]

    summary = f"Resolved {len(symbols)} definition(s) for '{name}'."
    debug = {
        "candidate_files": sorted(result.files_to_patterns.keys()),
        "rg_partial": result.partial,
    }
    return QueryResponse(summary=summary, primary=primary, related={}, debug=debug)


__all__ = ["handle"]
