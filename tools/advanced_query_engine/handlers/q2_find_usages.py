"""Reference discovery handler (Q2)."""

from __future__ import annotations

import re
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
    PATH_KIND_PROD,
    PATH_KIND_TEST,
    classify_path_kind,
    module_qname_from_path,
)
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


@dataclass(frozen=True)
class _RoleIndex:
    call_callees: list[Span]
    import_spans: list[Span]
    inherit_spans: list[Span]
    def_name_spans: list[Span]


def _role_queries(context: SearchContext) -> list[TreeSitterQuery]:
    queries: list[TreeSitterQuery] = []
    for pack_id in ("ts.python.calls", "ts.python.imports", "ts.python.defs"):
        try:
            query_text = context.query_catalog.tree_sitter_pack(pack_id)
        except ValueError:
            continue
        if query_text:
            queries.append(TreeSitterQuery(pack_id=pack_id, query_text=query_text))
    return queries


def _role_index_for_file(
    context: SearchContext,
    rel_path: str,
    queries: list[TreeSitterQuery],
    budget: QueryBudget,
) -> _RoleIndex:
    if not queries:
        return _RoleIndex(call_callees=[], import_spans=[], inherit_spans=[], def_name_spans=[])
    source_bytes = context.cache.read_bytes(rel_path)
    parsed = context.tree_sitter_parse(rel_path, "python")
    match_limit = budget.max_matches if budget.max_matches else 2000
    result = run_query_packs(
        TreeSitterRequest(
            language="python",
            source_bytes=source_bytes,
            path=rel_path,
            queries=queries,
            match_limit=match_limit,
            preview_limit=200,
            parsed=parsed,
        )
    )
    call_callees: list[Span] = []
    import_spans: list[Span] = []
    inherit_spans: list[Span] = []
    def_name_spans: list[Span] = []
    for cap in result.captures:
        if cap.capture_name == "call.callee":
            call_callees.append(cap.span)
            continue
        if cap.capture_name.startswith("import."):
            import_spans.append(cap.span)
            continue
        if cap.capture_name == "def.class.bases":
            inherit_spans.append(cap.span)
            continue
        if cap.capture_name in {"def.func.name", "def.class.name", "def.assign.name"}:
            def_name_spans.append(cap.span)
    return _RoleIndex(
        call_callees=call_callees,
        import_spans=import_spans,
        inherit_spans=inherit_spans,
        def_name_spans=def_name_spans,
    )


def _span_contains(span: Span, inner: Span) -> bool:
    return span.start_byte <= inner.start_byte and inner.end_byte <= span.end_byte


def _is_write_occurrence(line_text: str, name: str) -> bool:
    assign_pattern = rf"\b{re.escape(name)}\b\s*=(?!=)"
    annotation_pattern = rf"\b{re.escape(name)}\b\s*:"
    return bool(re.search(assign_pattern, line_text)) or bool(
        re.search(annotation_pattern, line_text)
    )


def _classify_match(
    name: str,
    span: Span,
    line_text: str,
    role_index: _RoleIndex,
) -> tuple[str, float]:
    role = "read"
    confidence = 0.5
    if any(_span_contains(def_span, span) for def_span in role_index.def_name_spans):
        role, confidence = "definition", 0.4
    elif any(_span_contains(base_span, span) for base_span in role_index.inherit_spans):
        role, confidence = "inherit", 0.9
    elif any(_span_contains(call_span, span) for call_span in role_index.call_callees):
        role, confidence = "call", 0.9
    elif any(_span_contains(import_span, span) for import_span in role_index.import_spans):
        role, confidence = "import", 0.8
    else:
        stripped = line_text.lstrip()
        if stripped.startswith(("import ", "from ")):
            role, confidence = "import", 0.7
        elif _is_write_occurrence(line_text, name):
            role, confidence = "write", 0.6
        elif re.search(rf"\b{re.escape(name)}\s*\(", line_text):
            role, confidence = "call", 0.7
    return role, confidence


def _role_weight(role: str) -> float:
    weights = {
        "call": 4.0,
        "import": 3.0,
        "inherit": 3.0,
        "write": 2.0,
        "read": 1.0,
        "definition": 0.5,
    }
    return weights.get(role, 1.0)


def _kind_weight(kind: str) -> float:
    weights = {
        PATH_KIND_PROD: 2.0,
        PATH_KIND_TEST: 1.0,
        PATH_KIND_EXAMPLE: 0.75,
        PATH_KIND_DOC: 0.5,
    }
    return weights.get(kind, 1.0)


def _override_candidates(
    name: str, context: SearchContext, candidate_files: list[str], budget: QueryBudget
) -> list[dict[str, object]]:
    overrides: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for rel_path in candidate_files:
        if budget.max_matches and len(overrides) >= budget.max_matches:
            break
        if not rel_path.endswith(".py"):
            continue
        try:
            def_index = context.def_index(rel_path)
        except (FileNotFoundError, ValueError):
            continue
        for record in def_index.by_name(name):
            if record.kind != "method":
                continue
            key = (record.qname, record.path)
            if key in seen:
                continue
            seen.add(key)
            overrides.append(
                {
                    "name": record.name,
                    "kind": record.kind,
                    "qname": record.qname,
                    "span": record.span.to_dict(),
                    "path": record.path,
                    "signature": record.signature,
                    "docstring": record.docstring,
                }
            )
            if budget.max_matches and len(overrides) >= budget.max_matches:
                break
    return overrides


@dataclass
class _RefSummary:
    records: list[dict[str, object]]
    role_counts: dict[str, int]
    kind_counts: dict[str, int]
    module_counts: dict[str, int]


def _build_reference_record(
    span: Span,
    context: SearchContext,
    role: str,
    confidence: float,
) -> dict[str, object]:
    snippet = build_snippet(
        SnippetRequest(
            source=context.cache.read_bytes(span.path),
            span=span,
            config=context.snippet_config,
            line_index=context.line_index(span.path),
        )
    )
    try:
        def_index = context.def_index(span.path)
        enclosing = def_index.enclosing_def(span.start_byte)
    except (FileNotFoundError, ValueError):
        enclosing = None
    record: dict[str, object] = {
        "path": span.path,
        "span": span.to_dict(),
        "snippet": snippet.to_dict(),
        "role": role,
        "confidence": confidence,
    }
    if enclosing is not None:
        record["enclosing"] = {
            "name": enclosing.name,
            "qname": enclosing.qname,
            "kind": enclosing.kind,
        }
    return record


def _iter_reference_records(
    name: str,
    matches: list[RpygrepMatch],
    context: SearchContext,
    budget: QueryBudget,
    role_queries: list[TreeSitterQuery],
) -> list[tuple[dict[str, object], str, str, str]]:
    seen: set[tuple[str, int, int, str]] = set()
    role_index_cache: dict[str, _RoleIndex] = {}
    records: list[tuple[dict[str, object], str, str, str]] = []
    for match in matches[: budget.max_matches]:
        span = _span_from_match(match, context)
        if span is None:
            continue
        role_index = role_index_cache.get(span.path)
        if role_index is None:
            role_index = _role_index_for_file(context, span.path, role_queries, budget)
            role_index_cache[span.path] = role_index
        role, confidence = _classify_match(name, span, match.line_text, role_index)
        if role == "definition":
            continue
        dedupe_key = (span.path, span.start_byte, span.end_byte, role)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        record = _build_reference_record(span, context, role, confidence)
        path_kind = classify_path_kind(span.path)
        module_name = module_qname_from_path(span.path)
        records.append((record, role, path_kind, module_name))
    return records


def _summarize_references(
    name: str,
    matches: list[RpygrepMatch],
    context: SearchContext,
    budget: QueryBudget,
    role_queries: list[TreeSitterQuery],
) -> _RefSummary:
    records = _iter_reference_records(name, matches, context, budget, role_queries)
    summary = _RefSummary(records=[], role_counts={}, kind_counts={}, module_counts={})
    for record, role, path_kind, module_name in records:
        summary.records.append(record)
        summary.role_counts[role] = summary.role_counts.get(role, 0) + 1
        summary.kind_counts[path_kind] = summary.kind_counts.get(path_kind, 0) + 1
        summary.module_counts[module_name] = summary.module_counts.get(module_name, 0) + 1
    return summary


def _rank_records(records: list[dict[str, object]]) -> None:
    records.sort(
        key=lambda item: (
            -(_role_weight(str(item.get("role"))) + _kind_weight(classify_path_kind(item["path"]))),
            item["path"],
            item["span"]["start_byte"],
        )
    )
    for idx, record in enumerate(records, start=1):
        record["rank"] = idx


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

    role_queries = _role_queries(context)

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

    summary = _summarize_references(name, result.matches, context, budget, role_queries)
    _rank_records(summary.records)

    candidate_files = sorted(result.files_to_patterns.keys())
    overrides = _override_candidates(name, context, candidate_files, budget)
    groups = {"by_module": summary.module_counts, "by_kind": summary.kind_counts}

    summary_text = f"Found {len(summary.records)} reference(s) for '{name}'."
    debug = {
        "rg_partial": result.partial,
        "rg_files": sorted(result.files_to_patterns.keys()),
        "role_counts": summary.role_counts,
    }
    related: dict[str, list[dict[str, object]]] = {"groups": [groups]}
    if overrides:
        related["overrides"] = overrides
    return QueryResponse(
        summary=summary_text,
        primary=summary.records,
        related=related,
        debug=debug,
    )


__all__ = ["handle"]
