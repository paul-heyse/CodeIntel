"""Impact slice handler (Q8)."""

from __future__ import annotations

from dataclasses import dataclass

from tools.advanced_query_engine.backends.treesitter_backend import (
    TreeSitterCapture,
    TreeSitterQuery,
    TreeSitterRequest,
    run_query_packs,
)
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import QueryBudget, QueryRequest, QueryResponse, Span
from tools.advanced_query_engine.handlers.q2_find_usages import handle as handle_find_usages
from tools.advanced_query_engine.handlers.q3_call_paths import handle as handle_call_paths
from tools.advanced_query_engine.util.semantic_helpers import (
    PATH_KIND_TEST,
    callee_label,
    classify_path_kind,
    package_prefix,
)
from tools.advanced_query_engine.util.snippets import SnippetRequest, build_snippet
from tools.advanced_query_engine.util.worktree import list_python_files


@dataclass(frozen=True)
class _CallEdge:
    caller_name: str | None
    caller_qname: str | None
    caller_kind: str | None
    caller_path: str | None
    callee_name: str
    callee_path: str | None
    call_span: Span


@dataclass(frozen=True)
class _SliceResult:
    outgoing: list[_CallEdge]
    incoming: list[_CallEdge]
    nodes: dict[str, dict[str, object]]
    crossings: list[dict[str, object]]
    packages: set[str]
    edge_total: int
    candidate_files: int


@dataclass(frozen=True)
class _ImpactPayload:
    related: dict[str, list[dict[str, object]]]
    edge_count: int
    test_count: int


@dataclass(frozen=True)
class _SliceInputs:
    target: str
    candidate_files: list[str]
    budget: QueryBudget
    query_text: str
    slice_params: tuple[int, int, int]


def _call_query(context: SearchContext) -> str:
    try:
        return context.query_catalog.tree_sitter_pack("ts.python.calls")
    except ValueError:
        return ""


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


def _within(outer: Span, inner: Span) -> bool:
    return outer.start_byte <= inner.start_byte and inner.end_byte <= outer.end_byte


def _select_closest(anchor: Span, candidates: list[Span], *, mode: str) -> Span | None:
    valid = [span for span in candidates if _within(anchor, span)]
    if not valid:
        return None
    if mode == "start":
        return min(valid, key=lambda span: abs(span.start_byte - anchor.start_byte))
    return min(valid, key=lambda span: abs(span.end_byte - anchor.end_byte))


def _call_edges_for_file(
    context: SearchContext,
    rel_path: str,
    captures: list[TreeSitterCapture],
) -> list[_CallEdge]:
    call_nodes = [cap for cap in captures if cap.capture_name == "call.node"]
    callee_caps = [cap for cap in captures if cap.capture_name == "call.callee"]
    callee_spans = [cap.span for cap in callee_caps]
    callee_texts = {cap.span: cap.text_preview for cap in callee_caps}
    edges: list[_CallEdge] = []
    try:
        def_index = context.def_index(rel_path)
    except (FileNotFoundError, ValueError):
        return edges
    def_map: dict[str, object] = {}
    for def_rec in def_index.defs:
        def_map.setdefault(def_rec.name, def_rec)
    for node in call_nodes:
        callee_span = _select_closest(node.span, callee_spans, mode="start")
        if callee_span is None:
            continue
        callee_text = callee_texts.get(callee_span) or ""
        if not callee_text:
            continue
        callee_name = callee_label(callee_text)
        caller = def_index.enclosing_def(node.span.start_byte)
        callee_rec = def_map.get(callee_name)
        edges.append(
            _CallEdge(
                caller_name=caller.name if caller is not None else None,
                caller_qname=caller.qname if caller is not None else None,
                caller_kind=caller.kind if caller is not None else None,
                caller_path=caller.path if caller is not None else None,
                callee_name=callee_name,
                callee_path=callee_rec.path if callee_rec is not None else None,
                call_span=node.span,
            )
        )
    return edges


def _slice_depth(value: object, default: int) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    return default


def _slice_options(options: dict[str, object]) -> tuple[int, int, int]:
    slice_cfg = options.get("slice")
    if isinstance(slice_cfg, dict):
        caller_depth = _slice_depth(slice_cfg.get("caller_depth"), 1)
        callee_depth = _slice_depth(slice_cfg.get("callee_depth"), 1)
    else:
        caller_depth = _slice_depth(options.get("caller_depth"), 1)
        callee_depth = _slice_depth(options.get("callee_depth"), 1)
    package_depth = _slice_depth(options.get("package_depth"), 1)
    return caller_depth, callee_depth, package_depth


def _build_edge_payload(edge: _CallEdge, context: SearchContext) -> dict[str, object]:
    snippet = build_snippet(
        SnippetRequest(
            source=context.cache.read_bytes(edge.call_span.path),
            span=edge.call_span,
            config=context.snippet_config,
            line_index=context.line_index(edge.call_span.path),
        )
    )
    return {
        "caller": {
            "name": edge.caller_name,
            "qname": edge.caller_qname,
            "kind": edge.caller_kind,
            "path": edge.caller_path,
        },
        "callee": {
            "name": edge.callee_name,
            "path": edge.callee_path,
        },
        "call_span": edge.call_span.to_dict(),
        "callsite": snippet.to_dict(),
    }


def _candidate_files(
    context: SearchContext, request: QueryRequest, budget: QueryBudget
) -> list[str]:
    files = list_python_files(
        context.repo_root, scope_paths=request.scope_paths, max_depth=budget.max_depth
    )
    if budget.max_files:
        return files[: budget.max_files]
    return files


def _collect_edges(
    context: SearchContext,
    candidate_files: list[str],
    query_text: str,
    budget: QueryBudget,
) -> list[_CallEdge]:
    edges: list[_CallEdge] = []
    for rel_path in candidate_files:
        if not rel_path.endswith(".py"):
            continue
        captures = _call_captures(context, rel_path, query_text, budget)
        if not captures:
            continue
        edges.extend(_call_edges_for_file(context, rel_path, captures))
        if budget.max_matches and len(edges) >= budget.max_matches:
            break
    return edges


def _edge_index(
    edges: list[_CallEdge],
) -> tuple[dict[str, list[_CallEdge]], dict[str, list[_CallEdge]]]:
    calls_out: dict[str, list[_CallEdge]] = {}
    calls_in: dict[str, list[_CallEdge]] = {}
    for edge in edges:
        if edge.caller_name:
            calls_out.setdefault(edge.caller_name, []).append(edge)
        calls_in.setdefault(edge.callee_name, []).append(edge)
    return calls_out, calls_in


def _traverse_outgoing(
    target: str, calls_out: dict[str, list[_CallEdge]], depth: int
) -> list[_CallEdge]:
    edges: list[_CallEdge] = []
    frontier = {target}
    for _ in range(depth):
        next_frontier: set[str] = set()
        for name in frontier:
            for edge in calls_out.get(name, []):
                edges.append(edge)
                next_frontier.add(edge.callee_name)
        frontier = next_frontier
    return edges


def _traverse_incoming(
    target: str, calls_in: dict[str, list[_CallEdge]], depth: int
) -> list[_CallEdge]:
    edges: list[_CallEdge] = []
    frontier = {target}
    for _ in range(depth):
        next_frontier: set[str] = set()
        for name in frontier:
            for edge in calls_in.get(name, []):
                edges.append(edge)
                if edge.caller_name:
                    next_frontier.add(edge.caller_name)
        frontier = next_frontier
    return edges


def _node_records(
    target: str, outgoing: list[_CallEdge], incoming: list[_CallEdge]
) -> dict[str, dict[str, object]]:
    nodes: dict[str, dict[str, object]] = {}
    for edge in outgoing + incoming:
        if edge.caller_name and edge.caller_name not in nodes:
            nodes[edge.caller_name] = {
                "name": edge.caller_name,
                "qname": edge.caller_qname,
                "kind": edge.caller_kind,
                "path": edge.caller_path,
            }
        if edge.callee_name not in nodes:
            nodes[edge.callee_name] = {
                "name": edge.callee_name,
                "path": edge.callee_path,
            }
    nodes.setdefault(target, {"name": target})
    return nodes


def _boundary_crossings(
    edges: list[_CallEdge], package_depth: int
) -> tuple[list[dict[str, object]], set[str]]:
    crossings: list[dict[str, object]] = []
    packages: set[str] = set()
    for edge in edges:
        caller_path = edge.caller_path
        callee_path = edge.callee_path
        if caller_path:
            packages.add(package_prefix(caller_path, depth=package_depth))
        if callee_path:
            packages.add(package_prefix(callee_path, depth=package_depth))
        if not caller_path or not callee_path:
            continue
        caller_pkg = package_prefix(caller_path, depth=package_depth)
        callee_pkg = package_prefix(callee_path, depth=package_depth)
        if caller_pkg and callee_pkg and caller_pkg != callee_pkg:
            crossings.append(
                {
                    "caller": caller_path,
                    "callee": callee_path,
                    "caller_package": caller_pkg,
                    "callee_package": callee_pkg,
                }
            )
    return crossings, packages


def _risk_summary(
    incoming: list[_CallEdge],
    packages: set[str],
    crossings: list[dict[str, object]],
    affected_tests: set[str],
) -> dict[str, object]:
    score = len(incoming) + len(packages) + 2 * len(crossings) + len(affected_tests)
    return {
        "score": score,
        "callers": len(incoming),
        "packages": len(packages),
        "public_crossings": len(crossings),
        "tests": len(affected_tests),
    }


def _affected_tests(refs: list[dict[str, object]]) -> set[str]:
    return {
        ref["path"]
        for ref in refs
        if isinstance(ref, dict)
        if classify_path_kind(ref.get("path", "")) == PATH_KIND_TEST
    }


def _compute_slice(
    context: SearchContext,
    inputs: _SliceInputs,
) -> _SliceResult:
    edges = _collect_edges(context, inputs.candidate_files, inputs.query_text, inputs.budget)
    calls_out, calls_in = _edge_index(edges)
    outgoing = _traverse_outgoing(inputs.target, calls_out, inputs.slice_params[1])
    incoming = _traverse_incoming(inputs.target, calls_in, inputs.slice_params[0])
    nodes = _node_records(inputs.target, outgoing, incoming)
    crossings, packages = _boundary_crossings(outgoing + incoming, inputs.slice_params[2])
    return _SliceResult(
        outgoing=outgoing,
        incoming=incoming,
        nodes=nodes,
        crossings=crossings,
        packages=packages,
        edge_total=len(edges),
        candidate_files=len(inputs.candidate_files),
    )


def _impact_payload(
    slice_result: _SliceResult,
    refs_response: QueryResponse,
    call_response: QueryResponse,
    context: SearchContext,
) -> _ImpactPayload:
    affected_tests = _affected_tests(refs_response.primary)
    risk = _risk_summary(
        slice_result.incoming,
        slice_result.packages,
        slice_result.crossings,
        affected_tests,
    )
    calls_in_payloads = call_response.related.get("calls_in", [])
    calls_out_payloads = call_response.related.get("calls_out", [])
    transitive_payloads = [
        _build_edge_payload(edge, context)
        for edge in slice_result.outgoing + slice_result.incoming
    ]
    related = {
        "edges": transitive_payloads,
        "calls_in": calls_in_payloads,
        "calls_out": calls_out_payloads,
        "references": refs_response.primary,
        "boundary_crossings": slice_result.crossings,
        "affected_tests": [{"path": path} for path in sorted(affected_tests)],
        "risk": [risk],
    }
    return _ImpactPayload(
        related=related,
        edge_count=len(transitive_payloads),
        test_count=len(affected_tests),
    )


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
    target = request.text.strip()
    if not target:
        return QueryResponse(
            summary="Empty symbol name; no results.",
            primary=[],
            related={},
            debug={"reason": "empty_symbol"},
        )

    budget = request.budget or context.default_budget
    if not isinstance(budget, QueryBudget):
        budget = QueryBudget()

    query_text = _call_query(context)
    if not query_text:
        return QueryResponse(
            summary="No tree-sitter call pack available.",
            primary=[],
            related={},
            debug={"reason": "no_calls"},
        )

    slice_params = _slice_options(request.options or {})
    candidate_files = _candidate_files(context, request, budget)
    slice_inputs = _SliceInputs(
        target=target,
        candidate_files=candidate_files,
        budget=budget,
        query_text=query_text,
        slice_params=slice_params,
    )
    slice_result = _compute_slice(context, slice_inputs)
    refs_response = handle_find_usages(request, context)
    call_response = handle_call_paths(request, context)
    payload = _impact_payload(slice_result, refs_response, call_response, context)

    summary = (
        f"Impact slice: {len(slice_result.nodes)} node(s), {payload.edge_count} edge(s), "
        f"{payload.test_count} test file(s) for '{target}'."
    )
    debug = {
        "refs_debug": refs_response.debug,
        "calls_debug": call_response.debug,
        "candidate_files": slice_result.candidate_files,
        "total_edges": slice_result.edge_total,
    }

    return QueryResponse(
        summary=summary,
        primary=list(slice_result.nodes.values()),
        related=payload.related,
        debug=debug,
    )


__all__ = ["handle"]
