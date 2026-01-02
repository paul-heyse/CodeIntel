"""Pattern and policy scan handler (Q4)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from tools.advanced_query_engine.backends.astgrep_backend import (
    run_rules_on_root,
    select_rules,
)
from tools.advanced_query_engine.backends.rpygrep_backend import (
    RpygrepMatch,
    RpygrepQuery,
    RpygrepResult,
    run_pattern_group,
)
from tools.advanced_query_engine.backends.treesitter_backend import (
    TreeSitterQuery,
    TreeSitterRequest,
    run_query_packs,
)
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import (
    EvidenceSnippet,
    MatchRecord,
    QueryBudget,
    QueryRequest,
    QueryResponse,
    Span,
)
from tools.advanced_query_engine.handlers.common import load_rpygrep_preset
from tools.advanced_query_engine.util.snippets import SnippetRequest, build_snippet
from tools.advanced_query_engine.util.worktree import list_python_files


def _fallback_pattern_group(pattern: str) -> dict[str, object]:
    return {
        "pattern_group_id": "rg.pattern.scan.fallback",
        "patterns": [
            {
                "pattern": pattern,
                "is_regex": True,
                "priority": 10,
            }
        ],
        "globs": ["**/*"],
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


def _iter_python_files(root: Path, scope_paths: list[str] | None, budget: QueryBudget) -> list[str]:
    paths = list_python_files(root, scope_paths=scope_paths, max_depth=budget.max_depth)
    if budget.max_files:
        return paths[: budget.max_files]
    return paths


@dataclass(frozen=True)
class _PatternScanInputs:
    request: QueryRequest
    context: SearchContext
    budget: QueryBudget


def _deadline(budget: QueryBudget) -> float | None:
    if budget.max_seconds is None:
        return None
    return time.monotonic() + float(budget.max_seconds)


def _budget_exhausted(budget: QueryBudget, deadline: float | None, count: int) -> bool:
    return (budget.max_matches > 0 and count >= budget.max_matches) or (
        deadline is not None and time.monotonic() >= deadline
    )


def _resolve_pattern_group(
    request: QueryRequest, context: SearchContext, options: dict[str, object]
) -> dict[str, object] | None:
    pattern_group_id = options.get("pattern_group_id")
    if pattern_group_id:
        return context.query_catalog.pattern_group(str(pattern_group_id))
    if not request.text:
        return None
    return _fallback_pattern_group(request.text)


def _rpygrep_record(match: RpygrepMatch, context: SearchContext) -> dict[str, object] | None:
    span = _span_from_match(match, context)
    if span is None:
        return None
    snippet = build_snippet(
        SnippetRequest(
            source=context.cache.read_bytes(span.path),
            span=span,
            config=context.snippet_config,
            line_index=context.line_index(span.path),
        )
    )
    record = MatchRecord(
        engine="rpygrep",
        path=span.path,
        span=span,
        pattern_id=match.pattern_id,
        snippet=snippet,
    )
    return record.to_dict()


def _scan_rpygrep(
    *,
    scan: _PatternScanInputs,
    preset: dict[str, object],
    pattern_group: dict[str, object],
) -> tuple[list[dict[str, object]], RpygrepResult]:
    rg_result = run_pattern_group(
        RpygrepQuery(
            repo_root=scan.context.repo_root,
            preset=preset,
            pattern_group=pattern_group,
            budget=scan.budget,
            scope_paths=scan.request.scope_paths,
            cache=scan.context.cache,
        )
    )
    records: list[dict[str, object]] = []
    for match in rg_result.matches[: scan.budget.max_matches]:
        record = _rpygrep_record(match, scan.context)
        if record is not None:
            records.append(record)
    return records, rg_result


def _collect_ast_matches(
    *,
    context: SearchContext,
    rules: list[object],
    candidate_files: list[str],
    budget: QueryBudget,
    deadline: float | None,
) -> tuple[list[object], bool]:
    matches: list[object] = []
    partial = False
    for rel_path in candidate_files:
        if _budget_exhausted(budget, deadline, len(matches)):
            partial = True
            break
        if not rel_path.endswith(".py"):
            continue
        try:
            root = context.ast_grep_root(rel_path)
        except (FileNotFoundError, ValueError):
            continue
        for match in run_rules_on_root(rel_path=rel_path, root=root, rules=rules):
            matches.append(match)
            if _budget_exhausted(budget, deadline, len(matches)):
                partial = True
                break
        if partial:
            break
    return matches, partial


def _ast_grep_records(
    *,
    scan: _PatternScanInputs,
    rg_result: RpygrepResult,
    ast_pack_id: object,
    rule_ids: object,
    deadline: float | None,
) -> tuple[list[dict[str, object]], bool]:
    if not ast_pack_id or not rule_ids:
        return [], False
    rule_pack = scan.context.query_catalog.ast_grep_rule_pack(str(ast_pack_id))
    rules = select_rules(rule_pack, [str(rule_id) for rule_id in rule_ids])
    candidate_files = sorted(rg_result.files_to_patterns.keys())
    if not candidate_files:
        candidate_files = _iter_python_files(
            scan.context.repo_root, scan.request.scope_paths, scan.budget
        )
    ast_matches, partial = _collect_ast_matches(
        context=scan.context,
        rules=rules,
        candidate_files=candidate_files,
        budget=scan.budget,
        deadline=deadline,
    )

    ast_records: list[dict[str, object]] = []
    for match in ast_matches:
        if _budget_exhausted(scan.budget, deadline, len(ast_records)):
            partial = True
            break
        line_index = scan.context.line_index(match.path)
        span = Span(
            path=match.path,
            start_byte=match.match_start,
            end_byte=match.match_end,
            **line_index.span_to_range(match.match_start, match.match_end),
        )
        source_bytes = scan.context.cache.read_bytes(match.path)
        snippet = build_snippet(
            SnippetRequest(
                source=source_bytes,
                span=span,
                config=scan.context.snippet_config,
                line_index=line_index,
            )
        )
        captures: dict[str, list[EvidenceSnippet]] = {
            name: [
                build_snippet(
                    SnippetRequest(
                        source=source_bytes,
                        span=Span(
                            path=match.path,
                            start_byte=value.start_byte,
                            end_byte=value.end_byte,
                        ),
                        config=scan.context.snippet_config,
                        line_index=line_index,
                    )
                )
                for value in values
            ]
            for name, values in match.captures.items()
        }
        record = MatchRecord(
            engine="ast_grep",
            path=match.path,
            span=span,
            rule_id=match.rule_id,
            snippet=snippet,
            captures=captures,
        )
        ast_records.append(record.to_dict())
    return ast_records, partial


def _tree_sitter_records(
    *,
    scan: _PatternScanInputs,
    ts_pack_id: object,
    deadline: float | None,
) -> tuple[list[dict[str, object]], bool]:
    if not ts_pack_id:
        return [], False
    try:
        query_text = scan.context.query_catalog.tree_sitter_pack(str(ts_pack_id))
    except ValueError:
        return [], False
    if not query_text:
        return [], False
    query = TreeSitterQuery(pack_id=str(ts_pack_id), query_text=query_text)
    ts_records: list[dict[str, object]] = []
    partial = False
    for rel_path in _iter_python_files(
        scan.context.repo_root, scan.request.scope_paths, scan.budget
    ):
        if _budget_exhausted(scan.budget, deadline, len(ts_records)):
            partial = True
            break
        source_bytes = scan.context.cache.read_bytes(rel_path)
        parsed = scan.context.tree_sitter_parse(rel_path, "python")
        result = run_query_packs(
            TreeSitterRequest(
                language="python",
                source_bytes=source_bytes,
                path=rel_path,
                queries=[query],
                match_limit=scan.budget.max_matches,
                preview_limit=200,
                parsed=parsed,
            )
        )
        for cap in result.captures:
            if _budget_exhausted(scan.budget, deadline, len(ts_records)):
                partial = True
                break
            ts_records.append(
                {
                    "capture": cap.capture_name,
                    "span": cap.span.to_dict(),
                    "preview": cap.text_preview,
                }
            )
        if partial:
            break
    return ts_records, partial


def handle(request: QueryRequest, context: SearchContext) -> QueryResponse:
    """Run a pattern or pack-driven scan.

    Parameters
    ----------
    request:
        Query request containing the pattern text and options.
    context:
        Search context providing indices and catalogs.

    Returns
    -------
    QueryResponse
        Query response with match records and related outputs.
    """
    budget = request.budget or context.default_budget
    if not isinstance(budget, QueryBudget):
        budget = QueryBudget()

    options = request.options or {}
    preset_id = (
        str(options.get("preset_id")) if options.get("preset_id") else "rg.default_interactive"
    )
    preset = load_rpygrep_preset(context.query_catalog, preset_id)

    pattern_group = _resolve_pattern_group(request, context, options)
    if pattern_group is None:
        return QueryResponse(
            summary="No pattern provided.",
            primary=[],
            related={},
            debug={"reason": "empty_pattern"},
        )

    scan = _PatternScanInputs(request=request, context=context, budget=budget)
    deadline = _deadline(budget)

    primary, rg_result = _scan_rpygrep(
        scan=scan,
        preset=preset,
        pattern_group=pattern_group,
    )

    related: dict[str, list[dict[str, object]]] = {}
    ast_records = _ast_grep_records(
        scan=scan,
        rg_result=rg_result,
        ast_pack_id=options.get("ast_grep_pack_id"),
        rule_ids=options.get("rule_ids"),
        deadline=deadline,
    )
    if ast_records[0]:
        related["ast_grep"] = ast_records[0]

    ts_records = _tree_sitter_records(
        scan=scan,
        ts_pack_id=options.get("tree_sitter_pack_id"),
        deadline=deadline,
    )
    if ts_records[0]:
        related["tree_sitter"] = ts_records[0]

    summary = f"Pattern scan produced {len(primary)} rpygrep matches."
    debug = {
        "rg_partial": rg_result.partial,
        "rg_files": sorted(rg_result.files_to_patterns.keys()),
        "ast_partial": ast_records[1],
        "ts_partial": ts_records[1],
    }
    return QueryResponse(summary=summary, primary=primary, related=related, debug=debug)


__all__ = ["handle"]
