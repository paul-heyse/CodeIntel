"""Precedent search handler (Q7)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tools.advanced_query_engine.backends.rpygrep_backend import (
    RpygrepQuery,
    run_pattern_group,
)
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import QueryBudget, QueryRequest, QueryResponse
from tools.advanced_query_engine.handlers.common import load_rpygrep_preset
from tools.advanced_query_engine.util.semantic_helpers import (
    extract_decorators,
    module_qname_from_path,
    parse_signature,
    tokenize_words,
)
from tools.advanced_query_engine.util.snippets import SnippetRequest, build_snippet
from tools.advanced_query_engine.util.worktree import list_python_files

if TYPE_CHECKING:
    from tools.advanced_query_engine.backends.libcst_backend import DefRecord
    from tools.advanced_query_engine.util.line_index import LineIndex


@dataclass(frozen=True)
class _DefFeatures:
    path: str
    name: str
    kind: str
    qname: str
    signature: str | None
    docstring: str | None
    decorators: set[str]
    param_count: int | None
    doc_tokens: set[str]


def _precedent_pattern_group(text: str) -> dict[str, object]:
    return {
        "pattern_group_id": "rg.precedent.search",
        "patterns": [
            {
                "pattern": text,
                "is_regex": True,
                "priority": 10,
            }
        ],
        "globs": ["**/*.py"],
    }


def _infer_name(query: str) -> str | None:
    text = query.strip()
    if not text:
        return None
    match = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    if match:
        return match.group(1)
    match = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text
    return None


def _feature_from_def(
    context: SearchContext,
    def_rec: DefRecord,
    line_index_cache: dict[str, LineIndex],
) -> _DefFeatures:
    path = def_rec.path
    if path not in line_index_cache:
        line_index_cache[path] = context.line_index(path)
    line_index = line_index_cache[path]
    source_bytes = context.cache.read_bytes(path)
    decorators = set(extract_decorators(source_bytes, line_index, def_rec.span))
    signature_info = parse_signature(def_rec.signature)
    param_count = None
    if signature_info is not None:
        param_count = (
            len(signature_info.positional)
            + len(signature_info.kwonly)
            + (1 if signature_info.vararg else 0)
            + (1 if signature_info.kwarg else 0)
        )
    return _DefFeatures(
        path=path,
        name=def_rec.name,
        kind=def_rec.kind,
        qname=def_rec.qname,
        signature=def_rec.signature,
        docstring=def_rec.docstring,
        decorators=decorators,
        param_count=param_count,
        doc_tokens=tokenize_words(def_rec.docstring),
    )


def _score_candidate(
    prototype: _DefFeatures | None,
    candidate: _DefFeatures,
    query_name: str | None,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if prototype is not None:
        if prototype.kind == candidate.kind:
            score += 1.0
            reasons.append("same_kind")
        if prototype.param_count is not None and candidate.param_count is not None:
            delta = abs(prototype.param_count - candidate.param_count)
            if delta == 0:
                score += 1.0
                reasons.append("same_arity")
            elif delta == 1:
                score += 0.5
                reasons.append("similar_arity")
        overlap = prototype.decorators.intersection(candidate.decorators)
        if overlap:
            score += 1.0
            reasons.append(f"decorators:{','.join(sorted(overlap))}")
        doc_overlap = prototype.doc_tokens.intersection(candidate.doc_tokens)
        if doc_overlap:
            score += 0.5
            reasons.append("doc_overlap")
    if query_name and candidate.name == query_name:
        score += 2.0
        reasons.append("same_name")
    elif query_name and query_name in candidate.name:
        score += 0.5
        reasons.append("name_contains")
    return score, reasons


def _prototype_def(
    name: str | None,
    context: SearchContext,
    budget: QueryBudget,
) -> _DefFeatures | None:
    if not name:
        return None
    preset = load_rpygrep_preset(context.query_catalog, "rg.default_interactive")
    pattern_group = {
        "pattern_group_id": f"rg.precedent.defs.{name}",
        "patterns": [
            {"pattern": rf"\bdef\s+{re.escape(name)}\b", "is_regex": True, "priority": 10},
            {
                "pattern": rf"\bclass\s+{re.escape(name)}\b",
                "is_regex": True,
                "priority": 9,
            },
        ],
        "globs": ["**/*.py"],
    }
    result = run_pattern_group(
        RpygrepQuery(
            repo_root=context.repo_root,
            preset=preset,
            pattern_group=pattern_group,
            budget=budget,
            scope_paths=None,
            cache=context.cache,
        )
    )
    line_index_cache: dict[str, LineIndex] = {}
    for rel_path in sorted(result.files_to_patterns.keys()):
        if not rel_path.endswith(".py"):
            continue
        try:
            def_index = context.def_index(rel_path)
        except (FileNotFoundError, ValueError):
            continue
        defs = def_index.by_name(name)
        if defs:
            return _feature_from_def(context, defs[0], line_index_cache)
    return None


def _candidate_files(
    rg_result: object,
    context: SearchContext,
    request: QueryRequest,
    budget: QueryBudget,
) -> list[str]:
    rg_files = sorted(getattr(rg_result, "files_to_patterns", {}).keys())
    if rg_files:
        return rg_files[: budget.max_files] if budget.max_files else rg_files
    files = list_python_files(
        context.repo_root, scope_paths=request.scope_paths, max_depth=budget.max_depth
    )
    return files[: budget.max_files] if budget.max_files else files


def _candidate_record(
    def_rec: DefRecord,
    context: SearchContext,
    score: float,
    reasons: list[str],
) -> dict[str, object]:
    snippet = build_snippet(
        SnippetRequest(
            source=context.cache.read_bytes(def_rec.path),
            span=def_rec.span,
            config=context.snippet_config,
            line_index=context.line_index(def_rec.path),
        )
    )
    return {
        "path": def_rec.path,
        "module": module_qname_from_path(def_rec.path),
        "span": def_rec.span.to_dict(),
        "snippet": snippet.to_dict(),
        "name": def_rec.name,
        "kind": def_rec.kind,
        "qname": def_rec.qname,
        "signature": def_rec.signature,
        "score": score,
        "why": reasons,
    }


def _build_candidates(
    candidate_files: list[str],
    context: SearchContext,
    prototype: _DefFeatures | None,
    query_name: str | None,
) -> list[dict[str, object]]:
    line_index_cache: dict[str, LineIndex] = {}
    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    for rel_path in candidate_files:
        try:
            def_index = context.def_index(rel_path)
        except (FileNotFoundError, ValueError):
            continue
        for def_rec in def_index.defs:
            key = (def_rec.path, def_rec.span.start_byte, def_rec.span.end_byte)
            if key in seen:
                continue
            seen.add(key)
            features = _feature_from_def(context, def_rec, line_index_cache)
            score, reasons = _score_candidate(prototype, features, query_name)
            if score <= 0:
                continue
            candidates.append(_candidate_record(def_rec, context, score, reasons))
    return candidates


def handle(request: QueryRequest, context: SearchContext) -> QueryResponse:
    """Find precedents similar to a pattern or symbol text.

    Parameters
    ----------
    request:
        Query request containing the search text.
    context:
        Search context providing indices and catalogs.

    Returns
    -------
    QueryResponse
        Query response with precedent candidates.
    """
    query = request.text.strip()
    if not query:
        return QueryResponse(
            summary="Empty precedent query; no results.",
            primary=[],
            related={},
            debug={"reason": "empty_query"},
        )

    budget = request.budget or context.default_budget
    if not isinstance(budget, QueryBudget):
        budget = QueryBudget()

    options = request.options or {}
    pattern_group = (
        context.query_catalog.pattern_group(str(options.get("pattern_group_id")))
        if options.get("pattern_group_id")
        else _precedent_pattern_group(query)
    )

    preset = load_rpygrep_preset(context.query_catalog, "rg.default_interactive")
    rg_result = run_pattern_group(
        RpygrepQuery(
            repo_root=context.repo_root,
            preset=preset,
            pattern_group=pattern_group,
            budget=budget,
            scope_paths=request.scope_paths,
            cache=context.cache,
        )
    )

    query_name = _infer_name(query)
    prototype = _prototype_def(query_name, context, budget)

    candidate_files = _candidate_files(rg_result, context, request, budget)
    candidates = _build_candidates(candidate_files, context, prototype, query_name)
    candidates.sort(key=lambda item: (-float(item["score"]), item["path"], item["name"]))
    limit = int(options.get("k")) if isinstance(options.get("k"), int) and options.get("k") else 5
    primary = candidates[:limit]

    summary = f"Found {len(primary)} precedent candidate(s) for '{query}'."
    debug = {
        "rg_partial": rg_result.partial,
        "rg_files": sorted(rg_result.files_to_patterns.keys()),
        "candidate_files": len(candidate_files),
    }
    return QueryResponse(summary=summary, primary=primary, related={}, debug=debug)


__all__ = ["handle"]
