"""Contract lookup handler (Q5)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from tools.advanced_query_engine.backends.rpygrep_backend import (
    RpygrepMatch,
    RpygrepQuery,
    run_pattern_group,
)
from tools.advanced_query_engine.backends.treesitter_backend import (
    TreeSitterQuery,
    TreeSitterRequest,
    run_query_packs,
)
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import QueryBudget, QueryRequest, QueryResponse, Span
from tools.advanced_query_engine.handlers.common import load_rpygrep_preset
from tools.advanced_query_engine.util.semantic_helpers import (
    PATH_KIND_DOC,
    PATH_KIND_EXAMPLE,
    PATH_KIND_TEST,
    classify_path_kind,
)
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
    }


@dataclass(frozen=True)
class _TestNode:
    name: str
    kind: str
    span: Span


@dataclass
class _TestRecord:
    name: str
    kind: str
    path: str
    span: Span
    evidence: list[dict[str, object]]


@dataclass
class _ContractCollection:
    tests: dict[tuple[str, str, int, int], _TestRecord]
    docs: list[dict[str, object]]
    examples: list[dict[str, object]]


@dataclass(frozen=True)
class _TestLookup:
    context: SearchContext
    budget: QueryBudget
    query_text: str
    cache: dict[str, list[_TestNode]]


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


def _deadline(budget: QueryBudget) -> float | None:
    if budget.max_seconds is None:
        return None
    return time.monotonic() + float(budget.max_seconds)


def _budget_exhausted(budget: QueryBudget, deadline: float | None, count: int) -> bool:
    return (budget.max_matches > 0 and count >= budget.max_matches) or (
        deadline is not None and time.monotonic() >= deadline
    )


def _tests_query(context: SearchContext) -> str:
    try:
        return context.query_catalog.tree_sitter_pack("ts.python.tests")
    except ValueError:
        return ""


def _test_nodes_for_file(
    context: SearchContext,
    rel_path: str,
    query_text: str,
    budget: QueryBudget,
) -> list[_TestNode]:
    if not query_text:
        return []
    source_bytes = context.cache.read_bytes(rel_path)
    parsed = context.tree_sitter_parse(rel_path, "python")
    match_limit = budget.max_matches if budget.max_matches else 500
    result = run_query_packs(
        TreeSitterRequest(
            language="python",
            source_bytes=source_bytes,
            path=rel_path,
            queries=[TreeSitterQuery(pack_id="ts.python.tests", query_text=query_text)],
            match_limit=match_limit,
            preview_limit=200,
            parsed=parsed,
        )
    )
    nodes: list[_TestNode] = []
    for cap in result.captures:
        if not cap.capture_name.endswith(".name"):
            continue
        name = cap.text_preview or ""
        if not name:
            continue
        kind = "function" if cap.capture_name.startswith("test.func") else "class"
        if cap.capture_name.startswith("test.method"):
            kind = "method"
        nodes.append(_TestNode(name=name, kind=kind, span=cap.span))
    nodes.sort(key=lambda node: (node.span.start_byte, node.span.end_byte, node.name))
    return nodes


def _find_enclosing_test(span: Span, nodes: list[_TestNode]) -> _TestNode | None:
    candidates = [
        node for node in nodes if node.span.start_byte <= span.start_byte < node.span.end_byte
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda node: node.span.end_byte - node.span.start_byte)


def _assert_lines(
    source_bytes: bytes,
    span: Span,
    context: SearchContext,
    *,
    max_assertions: int = 5,
) -> list[str]:
    line_index = context.line_index(span.path)
    start_line, _ = line_index.line_col(span.start_byte)
    end_line, _ = line_index.line_col(span.end_byte)
    lines: list[str] = []
    for line_no in range(start_line, end_line + 1):
        begin = line_index.line_start_byte(line_no)
        finish = line_index.line_start_byte(line_no + 1)
        text = source_bytes[begin:finish].decode("utf-8", errors="replace").strip()
        if not text:
            continue
        if "assert " in text or "pytest.raises" in text or ".assert" in text or "raises(" in text:
            lines.append(text)
        if len(lines) >= max_assertions:
            break
    return lines


def _match_payload(span: Span, context: SearchContext) -> dict[str, object]:
    snippet = build_snippet(
        SnippetRequest(
            source=context.cache.read_bytes(span.path),
            span=span,
            config=context.snippet_config,
            line_index=context.line_index(span.path),
        )
    )
    return {"path": span.path, "span": span.to_dict(), "snippet": snippet.to_dict()}


def _add_test_match(
    collection: _ContractCollection,
    span: Span,
    lookup: _TestLookup,
) -> None:
    nodes = lookup.cache.get(span.path)
    if nodes is None:
        nodes = _test_nodes_for_file(lookup.context, span.path, lookup.query_text, lookup.budget)
        lookup.cache[span.path] = nodes
    match_payload = _match_payload(span, lookup.context)
    test_node = _find_enclosing_test(span, nodes)
    if test_node is None:
        key = (span.path, "<module>", span.start_byte, span.end_byte)
        record = collection.tests.get(key)
        if record is None:
            record = _TestRecord(
                name="<module>",
                kind="module",
                path=span.path,
                span=span,
                evidence=[],
            )
            collection.tests[key] = record
        record.evidence.append(match_payload)
        return
    key = (
        span.path,
        test_node.name,
        test_node.span.start_byte,
        test_node.span.end_byte,
    )
    record = collection.tests.get(key)
    if record is None:
        record = _TestRecord(
            name=test_node.name,
            kind=test_node.kind,
            path=span.path,
            span=test_node.span,
            evidence=[],
        )
        collection.tests[key] = record
    record.evidence.append(match_payload)


def _collect_contracts(
    matches: list[RpygrepMatch],
    context: SearchContext,
    budget: QueryBudget,
    query_text: str,
) -> _ContractCollection:
    collection = _ContractCollection(tests={}, docs=[], examples=[])
    test_nodes_cache: dict[str, list[_TestNode]] = {}
    test_lookup = _TestLookup(
        context=context,
        budget=budget,
        query_text=query_text,
        cache=test_nodes_cache,
    )
    deadline = _deadline(budget)
    for match in matches[: budget.max_matches]:
        total = len(collection.tests) + len(collection.docs) + len(collection.examples)
        if _budget_exhausted(budget, deadline, total):
            break
        span = _span_from_match(match, context)
        if span is None:
            continue
        kind = classify_path_kind(span.path)
        if kind == PATH_KIND_DOC:
            collection.docs.append(_match_payload(span, context))
            continue
        if kind == PATH_KIND_EXAMPLE:
            collection.examples.append(_match_payload(span, context))
            continue
        if kind == PATH_KIND_TEST:
            _add_test_match(collection, span, test_lookup)
    return collection


def _build_test_records(
    collection: _ContractCollection, context: SearchContext
) -> list[dict[str, object]]:
    test_records: list[dict[str, object]] = []
    for record in collection.tests.values():
        source_bytes = context.cache.read_bytes(record.path)
        snippet = build_snippet(
            SnippetRequest(
                source=source_bytes,
                span=record.span,
                config=context.snippet_config,
                line_index=context.line_index(record.path),
            )
        )
        assertions = _assert_lines(source_bytes, record.span, context)
        test_records.append(
            {
                "test_name": record.name,
                "test_kind": record.kind,
                "path": record.path,
                "span": record.span.to_dict(),
                "snippet": snippet.to_dict(),
                "assertions": assertions,
                "references": record.evidence,
            }
        )
    return test_records


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

    test_query = _tests_query(context)
    collection = _collect_contracts(result.matches, context, budget, test_query)
    test_records = _build_test_records(collection, context)

    summary = (
        f"Found {len(test_records)} test(s), {len(collection.docs)} doc hit(s), "
        f"{len(collection.examples)} example hit(s) for '{name}'."
    )
    related: dict[str, list[dict[str, object]]] = {}
    if collection.docs:
        related["docs"] = collection.docs
    if collection.examples:
        related["examples"] = collection.examples
    debug = {
        "rg_files": sorted(result.files_to_patterns.keys()),
        "rg_partial": result.partial,
    }
    return QueryResponse(summary=summary, primary=test_records, related=related, debug=debug)


__all__ = ["handle"]
