"""Validation harness for advanced query engine queries."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from tools.advanced_query_engine.backends.astgrep_backend import run_rules_on_root, select_rules
from tools.advanced_query_engine.backends.rpygrep_backend import (
    RpygrepMatch,
    RpygrepQuery,
    run_pattern_group,
)
from tools.advanced_query_engine.backends.treesitter_backend import (
    TreeSitterCapture,
    TreeSitterQuery,
    TreeSitterRequest,
    run_query_packs,
)
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import (
    JSONValue,
    QueryBudget,
    QueryRequest,
    QueryResponse,
)
from tools.advanced_query_engine.handlers.common import load_rpygrep_preset
from tools.advanced_query_engine.packs.catalog import PackCatalog, build_pack_catalog
from tools.advanced_query_engine.packs.wiring_validation import resolve_pack_path
from tools.advanced_query_engine.service import RepoServiceOptions, SearchService
from tools.advanced_query_engine.util.snippets import SnippetConfig
from tools.advanced_query_engine.util.worktree import list_python_files

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseDefinition:
    """Defines a validation case and its default request values."""

    case_id: str
    query_type: str
    default_text: str
    default_options: dict[str, JSONValue] | None = None


@dataclass
class CaseReport:
    """Structured outcome for a validation case."""

    case_id: str
    query_type: str
    ok: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, JSONValue]

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable dict representation.

        Returns
        -------
        dict[str, JSONValue]
            Serialized case report payload.
        """
        return {
            "case_id": self.case_id,
            "query_type": self.query_type,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class ValidationContext:
    """Cached context for validators."""

    repo_root: Path
    query_catalog: PackCatalog
    wiring_catalog: PackCatalog
    search_context: SearchContext
    python_files: list[str]
    budget: QueryBudget
    scope_paths: list[str] | None


CaseValidator = Callable[
    [ValidationContext, QueryRequest, QueryResponse, dict[str, QueryResponse]],
    CaseReport,
]


DEFAULT_BUDGET = QueryBudget(
    max_files=5000,
    max_matches=20000,
    max_depth=0,
    max_seconds=None,
    context_lines=1,
)


CASE_DEFINITIONS: dict[str, CaseDefinition] = {
    "symbol_resolve": CaseDefinition(
        case_id="symbol_resolve",
        query_type="symbol.resolve",
        default_text="compose_runtime",
    ),
    "find_usages": CaseDefinition(
        case_id="find_usages",
        query_type="refs.find",
        default_text="compose_runtime",
    ),
    "call_paths": CaseDefinition(
        case_id="call_paths",
        query_type="callgraph.slice",
        default_text="compose_runtime",
    ),
    "pattern_scan": CaseDefinition(
        case_id="pattern_scan",
        query_type="pattern.scan",
        default_text="def ",
        default_options={"preset_id": "rg.audit_deterministic"},
    ),
    "contract_lookup": CaseDefinition(
        case_id="contract_lookup",
        query_type="contract.lookup",
        default_text="compose_runtime",
    ),
    "wiring_map": CaseDefinition(
        case_id="wiring_map",
        query_type="wiring.map",
        default_text="",
        default_options={"pack_ids": ["wire.python.env"]},
    ),
    "precedent_search": CaseDefinition(
        case_id="precedent_search",
        query_type="precedent.search",
        default_text="compose_runtime",
    ),
    "impact_slice": CaseDefinition(
        case_id="impact_slice",
        query_type="impact.slice",
        default_text="compose_runtime",
        default_options={"include_wiring": False},
    ),
}


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate advanced query engine outputs with independent checks."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        help="Repo-relative path to constrain scanning; defaults to src and tests.",
    )
    parser.add_argument("--max-files", type=int, default=DEFAULT_BUDGET.max_files)
    parser.add_argument("--max-matches", type=int, default=DEFAULT_BUDGET.max_matches)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_BUDGET.max_depth)
    parser.add_argument("--context-lines", type=int, default=DEFAULT_BUDGET.context_lines)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        default=None,
        help="Run a specific case id (repeatable).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("build/advanced_query_engine/validation_report.json"),
    )
    return parser.parse_args(list(argv))


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _case_overrides(config: dict[str, object], case_id: str) -> dict[str, object]:
    cases = config.get("cases")
    if not isinstance(cases, dict):
        return {}
    overrides = cases.get(case_id)
    return overrides if isinstance(overrides, dict) else {}


def _coerce_json_value(value: object) -> tuple[JSONValue, bool]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value, True
    if isinstance(value, list):
        items: list[JSONValue] = []
        ok = True
        for entry in value:
            converted, entry_ok = _coerce_json_value(entry)
            if not entry_ok:
                ok = False
                break
            items.append(converted)
        return (items if ok else None), ok
    if isinstance(value, dict):
        payload: dict[str, JSONValue] = {}
        ok = True
        for key, entry in value.items():
            if not isinstance(key, str):
                ok = False
                break
            converted, entry_ok = _coerce_json_value(entry)
            if not entry_ok:
                ok = False
                break
            payload[key] = converted
        return (payload if ok else None), ok
    return None, False


def _merge_options(
    default_opts: dict[str, JSONValue] | None,
    override_opts: object,
) -> dict[str, JSONValue] | None:
    if default_opts is None and override_opts is None:
        return None
    merged: dict[str, JSONValue] = dict(default_opts or {})
    if isinstance(override_opts, dict):
        for key, value in override_opts.items():
            converted, ok = _coerce_json_value(value)
            if not ok:
                continue
            merged[str(key)] = converted
    return merged or None


def _build_request(
    *,
    definition: CaseDefinition,
    repo_root: Path,
    budget: QueryBudget,
    scope_paths: list[str] | None,
    config: dict[str, object],
) -> QueryRequest:
    overrides = _case_overrides(config, definition.case_id)
    text_override = overrides.get("text")
    text_value = definition.default_text
    if isinstance(text_override, str):
        text_value = text_override
    options = _merge_options(definition.default_options, overrides.get("options"))
    return QueryRequest(
        type=definition.query_type,
        text=text_value,
        repo_root=str(repo_root),
        scope_paths=scope_paths,
        budget=budget,
        options=options,
    )


def _span_key(span: dict[str, JSONValue] | None) -> tuple[str, int, int] | None:
    if span is None:
        return None
    path = span.get("path")
    start = span.get("start_byte")
    end = span.get("end_byte")
    if not isinstance(path, str) or not isinstance(start, int) or not isinstance(end, int):
        return None
    return (path, start, end)


def _record_span_key(record: dict[str, JSONValue]) -> tuple[str, int, int] | None:
    span = record.get("span")
    if not isinstance(span, dict):
        return None
    return _span_key(span)


def _symbol_key(record: dict[str, JSONValue]) -> tuple[str, int, int, str] | None:
    span = record.get("def_span")
    kind = record.get("kind")
    if not isinstance(span, dict) or not isinstance(kind, str):
        return None
    span_key = _span_key(span)
    if span_key is None:
        return None
    return (*span_key, kind)


def _call_span_key(record: dict[str, JSONValue]) -> tuple[str, int, int] | None:
    span = record.get("call_span")
    if not isinstance(span, dict):
        return None
    return _span_key(span)


def _rpygrep_span_key(match: RpygrepMatch, context: SearchContext) -> tuple[str, int, int] | None:
    if match.span is not None:
        return (match.span.path, match.span.start_byte, match.span.end_byte)
    try:
        index = context.line_index(match.path)
    except FileNotFoundError:
        return None
    line_start = index.line_start_byte(match.line_number)
    start = line_start + match.submatch_start
    end = line_start + match.submatch_end
    return (match.path, start, end)


def _callee_name(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return ""
    return expr.split(".")[-1]


def _libcst_defs(
    context: SearchContext,
    python_files: Iterable[str],
    name: str,
) -> set[tuple[str, int, int, str]]:
    defs: set[tuple[str, int, int, str]] = set()
    for rel_path in python_files:
        try:
            index = context.def_index(rel_path)
        except (FileNotFoundError, ValueError):
            continue
        for record in index.by_name(name):
            defs.add((record.path, record.span.start_byte, record.span.end_byte, record.kind))
    return defs


def _tree_sitter_matches(
    context: SearchContext,
    python_files: Iterable[str],
    *,
    pack_id: str,
    query_text: str,
    budget: QueryBudget,
) -> tuple[list[TreeSitterCapture], bool, list[str]]:
    captures: list[TreeSitterCapture] = []
    warnings: list[str] = []
    partial = False
    query = TreeSitterQuery(pack_id=pack_id, query_text=query_text)
    for rel_path in python_files:
        if budget.max_matches and len(captures) >= budget.max_matches:
            partial = True
            break
        source_bytes = context.cache.read_bytes(rel_path)
        parsed = context.tree_sitter_parse(rel_path, "python")
        result = run_query_packs(
            TreeSitterRequest(
                language="python",
                source_bytes=source_bytes,
                path=rel_path,
                queries=[query],
                match_limit=budget.max_matches,
                preview_limit=200,
                parsed=parsed,
            )
        )
        captures.extend(result.captures)
        warnings.extend(result.warnings)
        if not result.parse_ok:
            partial = True
    return captures, partial, warnings


def _rpygrep_matches(
    context: ValidationContext,
    *,
    preset_id: str,
    pattern_group: dict[str, object],
) -> tuple[list[RpygrepMatch], bool]:
    preset = load_rpygrep_preset(context.query_catalog, preset_id)
    result = run_pattern_group(
        RpygrepQuery(
            repo_root=context.repo_root,
            preset=preset,
            pattern_group=pattern_group,
            budget=context.budget,
            scope_paths=context.scope_paths,
            cache=context.search_context.cache,
        )
    )
    return result.matches, result.partial


def _pattern_group_for_text(pattern: str) -> dict[str, object]:
    return {
        "pattern_group_id": "rg.pattern.scan.fallback",
        "patterns": [{"pattern": pattern, "is_regex": True, "priority": 10}],
        "globs": ["**/*"],
    }


def _call_query_text(catalog: PackCatalog) -> str | None:
    try:
        return catalog.tree_sitter_pack("ts.python.calls")
    except ValueError:
        return None


def _expected_calls_in(
    context: ValidationContext,
    *,
    query_text: str,
    symbol: str,
) -> tuple[set[tuple[str, int, int]], bool, list[str]]:
    captures, partial, warnings = _tree_sitter_matches(
        context.search_context,
        context.python_files,
        pack_id="ts.python.calls",
        query_text=query_text,
        budget=context.budget,
    )
    expected = {
        (cap.span.path, cap.span.start_byte, cap.span.end_byte)
        for cap in captures
        if cap.capture_name == "call.callee"
        if _callee_name(cap.text_preview or "") == symbol
    }
    return expected, partial, warnings


def _engine_calls_in(response: QueryResponse) -> set[tuple[str, int, int]]:
    calls_in = response.related.get("calls_in", [])
    return {
        key
        for record in calls_in
        if isinstance(record, dict)
        if (key := _call_span_key(record)) is not None
    }


def _calls_out_of_bounds(
    context: ValidationContext,
    *,
    symbol: str,
    calls_out: list[object],
) -> int:
    def_spans = _libcst_defs(context.search_context, context.python_files, symbol)
    defs_by_path: dict[str, list[tuple[int, int]]] = {}
    for path, start, end, _kind in def_spans:
        defs_by_path.setdefault(path, []).append((start, end))
    out_of_bounds = 0
    for record in calls_out:
        if not isinstance(record, dict):
            continue
        span_key = _call_span_key(record)
        if span_key is None:
            continue
        path, start, end = span_key
        ranges = defs_by_path.get(path, [])
        if not any(start >= r_start and end <= r_end for r_start, r_end in ranges):
            out_of_bounds += 1
    return out_of_bounds


def validate_symbol_resolve(
    context: ValidationContext,
    request: QueryRequest,
    response: QueryResponse,
    _responses: dict[str, QueryResponse],
) -> CaseReport:
    """Validate symbol resolution results.

    Parameters
    ----------
    context:
        Validation context with cached indices.
    request:
        Query request used for the run.
    response:
        Query response produced by the engine.
    _responses:
        Prior query responses (unused).

    Returns
    -------
    CaseReport
        Validation report for the symbol resolution case.
    """
    expected = _libcst_defs(context.search_context, context.python_files, request.text)
    engine_defs = {
        key
        for record in response.primary
        if isinstance(record, dict)
        if (key := _symbol_key(record)) is not None
    }
    missing = expected - engine_defs
    extra = engine_defs - expected

    errors: list[str] = []
    warnings: list[str] = []
    if extra:
        errors.append("Engine produced definitions not found by LibCST index.")
    if missing:
        if response.debug.get("rg_partial"):
            warnings.append("Missing definitions due to partial rpygrep scan.")
        else:
            errors.append("LibCST definitions missing from engine results.")
    if not expected:
        warnings.append("LibCST definition scan returned no matches for the symbol.")

    metrics = {
        "expected_defs": len(expected),
        "engine_defs": len(engine_defs),
        "missing_defs": len(missing),
        "extra_defs": len(extra),
        "rg_partial": bool(response.debug.get("rg_partial")),
    }
    ok = not errors
    return CaseReport(
        case_id="symbol_resolve",
        query_type=request.type,
        ok=ok,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def validate_find_usages(
    context: ValidationContext,
    request: QueryRequest,
    response: QueryResponse,
    _responses: dict[str, QueryResponse],
) -> CaseReport:
    """Validate symbol usage results with tree-sitter captures.

    Parameters
    ----------
    context:
        Validation context with cached indices.
    request:
        Query request used for the run.
    response:
        Query response produced by the engine.
    _responses:
        Prior query responses (unused).

    Returns
    -------
    CaseReport
        Validation report for the usage lookup case.
    """
    query_text = "(identifier) @id\n(attribute attribute: (identifier) @id)"
    captures, partial, warnings = _tree_sitter_matches(
        context.search_context,
        context.python_files,
        pack_id="ts.python.identifiers",
        query_text=query_text,
        budget=context.budget,
    )
    expected = {
        (cap.span.path, cap.span.start_byte, cap.span.end_byte)
        for cap in captures
        if cap.capture_name == "id" and cap.text_preview == request.text
    }
    engine_spans = {
        key
        for record in response.primary
        if isinstance(record, dict)
        if (key := _record_span_key(record)) is not None
    }
    missing = expected - engine_spans

    errors: list[str] = []
    warn_list = warnings[:]
    if missing:
        if response.debug.get("rg_partial") or partial:
            warn_list.append("Missing usage spans may be due to partial scans.")
        else:
            errors.append("Engine missed identifier usages found by tree-sitter.")

    metrics = {
        "expected_usages": len(expected),
        "engine_usages": len(engine_spans),
        "missing_usages": len(missing),
        "ts_partial": partial,
        "rg_partial": bool(response.debug.get("rg_partial")),
    }
    ok = not errors
    return CaseReport(
        case_id="find_usages",
        query_type=request.type,
        ok=ok,
        errors=errors,
        warnings=warn_list,
        metrics=metrics,
    )


def validate_call_paths(
    context: ValidationContext,
    request: QueryRequest,
    response: QueryResponse,
    _responses: dict[str, QueryResponse],
) -> CaseReport:
    """Validate callgraph results using tree-sitter call captures.

    Parameters
    ----------
    context:
        Validation context with cached indices.
    request:
        Query request used for the run.
    response:
        Query response produced by the engine.
    _responses:
        Prior query responses (unused).

    Returns
    -------
    CaseReport
        Validation report for the callgraph slice case.
    """
    query_text = _call_query_text(context.query_catalog)
    if query_text is None:
        return CaseReport(
            case_id="call_paths",
            query_type=request.type,
            ok=False,
            errors=["Missing tree-sitter call pack (ts.python.calls)."],
            warnings=[],
            metrics={},
        )
    expected_calls, partial, warnings = _expected_calls_in(
        context,
        query_text=query_text,
        symbol=request.text,
    )
    engine_calls = _engine_calls_in(response)
    missing = expected_calls - engine_calls

    calls_out = response.related.get("calls_out", [])
    out_of_bounds = _calls_out_of_bounds(
        context,
        symbol=request.text,
        calls_out=calls_out,
    )

    errors: list[str] = []
    warn_list = warnings[:]
    if missing:
        if response.debug.get("rg_partial") or partial:
            warn_list.append("Missing call spans may be due to partial scans.")
        else:
            errors.append("Engine missed call sites found by tree-sitter.")
    if out_of_bounds:
        errors.append("Outgoing call spans are not within the target definition spans.")

    metrics = {
        "expected_calls_in": len(expected_calls),
        "engine_calls_in": len(engine_calls),
        "missing_calls_in": len(missing),
        "calls_out": len(calls_out),
        "calls_out_of_bounds": out_of_bounds,
        "ts_partial": partial,
        "rg_partial": bool(response.debug.get("rg_partial")),
    }
    ok = not errors
    return CaseReport(
        case_id="call_paths",
        query_type=request.type,
        ok=ok,
        errors=errors,
        warnings=warn_list,
        metrics=metrics,
    )


def validate_pattern_scan(
    context: ValidationContext,
    request: QueryRequest,
    response: QueryResponse,
    _responses: dict[str, QueryResponse],
) -> CaseReport:
    """Validate pattern scan results against rpygrep output.

    Parameters
    ----------
    context:
        Validation context with cached indices.
    request:
        Query request used for the run.
    response:
        Query response produced by the engine.
    _responses:
        Prior query responses (unused).

    Returns
    -------
    CaseReport
        Validation report for the pattern scan case.
    """
    options = request.options or {}
    preset_id = str(options.get("preset_id") or "rg.default_interactive")
    pattern_group_id = options.get("pattern_group_id")
    if pattern_group_id:
        pattern_group = context.query_catalog.pattern_group(str(pattern_group_id))
    else:
        pattern_group = _pattern_group_for_text(request.text)
    matches, partial = _rpygrep_matches(
        context,
        preset_id=preset_id,
        pattern_group=pattern_group,
    )
    expected = {
        key
        for match in matches
        if (key := _rpygrep_span_key(match, context.search_context)) is not None
    }
    engine_spans = {
        key
        for record in response.primary
        if isinstance(record, dict)
        if (key := _record_span_key(record)) is not None
    }
    missing = expected - engine_spans
    extra = engine_spans - expected

    errors: list[str] = []
    warnings: list[str] = []
    if extra:
        errors.append("Engine returned pattern matches not present in rpygrep output.")
    if missing:
        if response.debug.get("rg_partial") or partial:
            warnings.append("Missing pattern matches may be due to partial scans.")
        else:
            errors.append("Engine missed rpygrep pattern matches.")

    metrics = {
        "expected_matches": len(expected),
        "engine_matches": len(engine_spans),
        "missing_matches": len(missing),
        "extra_matches": len(extra),
        "rg_partial": bool(response.debug.get("rg_partial")),
    }
    ok = not errors
    return CaseReport(
        case_id="pattern_scan",
        query_type=request.type,
        ok=ok,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def validate_contract_lookup(
    context: ValidationContext,
    request: QueryRequest,
    response: QueryResponse,
    _responses: dict[str, QueryResponse],
) -> CaseReport:
    """Validate contract lookup results against rpygrep output.

    Parameters
    ----------
    context:
        Validation context with cached indices.
    request:
        Query request used for the run.
    response:
        Query response produced by the engine.
    _responses:
        Prior query responses (unused).

    Returns
    -------
    CaseReport
        Validation report for the contract lookup case.
    """
    escaped = re.escape(request.text)
    pattern_group = {
        "pattern_group_id": f"rg.contract.lookup.{request.text}",
        "patterns": [{"pattern": rf"\b{escaped}\b", "is_regex": True, "priority": 10}],
        "globs": [
            "**/tests/**/*.py",
            "**/test_*.py",
            "**/*_test.py",
            "**/docs/**/*.md",
            "**/README*",
            "**/examples/**/*",
        ],
    }
    matches, partial = _rpygrep_matches(
        context,
        preset_id="rg.default_interactive",
        pattern_group=pattern_group,
    )
    expected = {
        key
        for match in matches
        if (key := _rpygrep_span_key(match, context.search_context)) is not None
    }
    engine_spans = {
        key
        for record in response.primary
        if isinstance(record, dict)
        if (key := _record_span_key(record)) is not None
    }
    missing = expected - engine_spans
    extra = engine_spans - expected

    errors: list[str] = []
    warnings: list[str] = []
    if extra:
        errors.append("Engine returned contract matches not present in rpygrep output.")
    if missing:
        if response.debug.get("rg_partial") or partial:
            warnings.append("Missing contract matches may be due to partial scans.")
        else:
            errors.append("Engine missed contract matches from rpygrep output.")

    metrics = {
        "expected_matches": len(expected),
        "engine_matches": len(engine_spans),
        "missing_matches": len(missing),
        "extra_matches": len(extra),
        "rg_partial": bool(response.debug.get("rg_partial")),
    }
    ok = not errors
    return CaseReport(
        case_id="contract_lookup",
        query_type=request.type,
        ok=ok,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def _ast_grep_matches(
    context: ValidationContext,
    *,
    rules: list[object],
    match_limit: int,
    candidate_files: list[str] | None,
) -> tuple[set[tuple[str, str, int, int]], bool]:
    matches: set[tuple[str, str, int, int]] = set()
    partial = False
    paths = candidate_files if candidate_files is not None else context.python_files
    for rel_path in paths:
        if match_limit and len(matches) >= match_limit:
            partial = True
            break
        try:
            root = context.search_context.ast_grep_root(rel_path)
        except (FileNotFoundError, ValueError):
            continue
        for match in run_rules_on_root(rel_path=rel_path, root=root, rules=rules):
            matches.add((match.path, match.rule_id, match.match_start, match.match_end))
            if match_limit and len(matches) >= match_limit:
                partial = True
                break
        if partial:
            break
    return matches, partial


def _pack_id_list(request: QueryRequest, catalog: PackCatalog) -> list[str]:
    options = request.options or {}
    pack_ids = options.get("pack_ids")
    if isinstance(pack_ids, str):
        return [pack_ids]
    if isinstance(pack_ids, list):
        return [str(item) for item in pack_ids]
    return list(catalog.wiring_packs.keys())


def _pack_candidate_files(response: QueryResponse) -> dict[str, list[str]]:
    by_pack = response.related.get("by_pack")
    if not isinstance(by_pack, list):
        return {}
    candidates: dict[str, list[str]] = {}
    for entry in by_pack:
        if not isinstance(entry, dict):
            continue
        pack_id = entry.get("pack_id")
        debug = entry.get("debug")
        if not isinstance(pack_id, str) or not isinstance(debug, dict):
            continue
        rg_files = debug.get("rg_files")
        if isinstance(rg_files, list) and all(isinstance(item, str) for item in rg_files):
            candidates[pack_id] = rg_files
    return candidates


def _ast_rules_for_pack(context: ValidationContext, pack_id: str) -> list[object]:
    pack = context.wiring_catalog.wiring_pack(pack_id)
    stages = pack.get("stages") or []
    stage_defs = [stage for stage in stages if isinstance(stage, dict)]
    rules: list[object] = []
    for stage in stage_defs:
        if stage.get("engine") != "ast_grep":
            continue
        rules_file = stage.get("rules_file")
        if not isinstance(rules_file, str):
            continue
        rules_path = resolve_pack_path(context.wiring_catalog.root, rules_file)
        if rules_path.suffix.lower() in {".yaml", ".yml"}:
            rule_pack = context.wiring_catalog.load_yaml(rules_path)
        else:
            rule_pack = context.wiring_catalog.load_json(rules_path)
        rule_ids = [str(rule_id) for rule_id in stage.get("rule_ids") or []]
        rules.extend(select_rules(rule_pack, rule_ids))
    return rules


def _expected_wiring_matches(
    context: ValidationContext,
    pack_ids: list[str],
    candidate_map: dict[str, list[str]],
) -> tuple[set[tuple[str, str, int, int]], bool, list[str]]:
    expected_matches: set[tuple[str, str, int, int]] = set()
    partial = False
    warnings: list[str] = []
    for pack_id in pack_ids:
        rules = _ast_rules_for_pack(context, pack_id)
        if not rules:
            continue
        candidate_files = candidate_map.get(pack_id)
        if candidate_files is None:
            warnings.append(f"Missing rg_files candidates for pack '{pack_id}'.")
            continue
        matches, pack_partial = _ast_grep_matches(
            context,
            rules=rules,
            match_limit=context.budget.max_matches,
            candidate_files=candidate_files,
        )
        expected_matches.update(matches)
        partial = partial or pack_partial
    return expected_matches, partial, warnings


def _engine_wiring_matches(response: QueryResponse) -> tuple[set[tuple[str, str, int, int]], int]:
    engine_edges = [edge for edge in response.primary if isinstance(edge, dict)]
    engine_match_keys: set[tuple[str, str, int, int]] = set()
    missing_entry_keys = 0
    for edge in engine_edges:
        match = edge.get("match")
        if not isinstance(match, dict):
            continue
        rule_id = match.get("rule_id")
        match_span = match.get("match_span")
        if not isinstance(rule_id, str) or not isinstance(match_span, dict):
            continue
        span_key = _span_key(match_span)
        if span_key is None:
            continue
        path, start, end = span_key
        engine_match_keys.add((path, rule_id, start, end))
        entry_key = edge.get("entry_key")
        if isinstance(entry_key, str) and "<missing:" in entry_key:
            missing_entry_keys += 1
    return engine_match_keys, missing_entry_keys


def _wiring_validation_flags(response: QueryResponse) -> tuple[bool, bool]:
    validation = response.debug.get("validation")
    if not isinstance(validation, dict):
        return False, False
    has_errors = bool(validation.get("errors"))
    has_warnings = bool(validation.get("warnings"))
    return has_errors, has_warnings


def validate_wiring_map(
    context: ValidationContext,
    request: QueryRequest,
    response: QueryResponse,
    _responses: dict[str, QueryResponse],
) -> CaseReport:
    """Validate wiring map output against ast-grep matches.

    Parameters
    ----------
    context:
        Validation context with cached indices.
    request:
        Query request used for the run.
    response:
        Query response produced by the engine.
    _responses:
        Prior query responses (unused).

    Returns
    -------
    CaseReport
        Validation report for the wiring map case.
    """
    pack_ids = _pack_id_list(request, context.wiring_catalog)
    candidate_map = _pack_candidate_files(response)
    expected_matches, ast_partial, candidate_warnings = _expected_wiring_matches(
        context,
        pack_ids,
        candidate_map,
    )
    engine_match_keys, missing_entry_keys = _engine_wiring_matches(response)
    missing = engine_match_keys - expected_matches
    extra = expected_matches - engine_match_keys

    errors: list[str] = []
    warnings = list(candidate_warnings)
    if missing:
        if response.debug.get("budget_exhausted") or ast_partial:
            warnings.append("Missing wiring matches may be due to partial AST scans.")
        else:
            errors.append("Wiring edges reference AST matches not found in validation scan.")
    if missing_entry_keys:
        errors.append("Wiring edges include entry_key placeholders.")
    if extra:
        if response.debug.get("budget_exhausted") or ast_partial:
            warnings.append("AST matches without edges may be due to budgets.")
        else:
            warnings.append("AST matches produced no wiring edges.")

    has_errors, has_warnings = _wiring_validation_flags(response)
    if has_errors:
        errors.append("Wiring pack validation reported errors.")
    if has_warnings:
        warnings.append("Wiring pack validation reported warnings.")

    metrics = {
        "expected_matches": len(expected_matches),
        "engine_edges": len(response.primary),
        "missing_edges": len(missing),
        "extra_matches": len(extra),
        "entry_key_placeholders": missing_entry_keys,
        "ast_partial": ast_partial,
        "budget_exhausted": bool(response.debug.get("budget_exhausted")),
    }
    ok = not errors
    return CaseReport(
        case_id="wiring_map",
        query_type=request.type,
        ok=ok,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def validate_precedent_search(
    context: ValidationContext,
    request: QueryRequest,
    response: QueryResponse,
    _responses: dict[str, QueryResponse],
) -> CaseReport:
    """Validate precedent search results for basic structure.

    Parameters
    ----------
    context:
        Validation context with cached indices.
    request:
        Query request used for the run.
    response:
        Query response produced by the engine.
    _responses:
        Prior query responses (unused).

    Returns
    -------
    CaseReport
        Validation report for the precedent search case.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not response.primary:
        warnings.append("Engine returned no precedent candidates.")
    scores: list[float] = []
    for record in response.primary:
        if not isinstance(record, dict):
            errors.append("Precedent record is not a dict.")
            continue
        span = record.get("span")
        if not isinstance(span, dict):
            errors.append("Precedent record missing span.")
            continue
        if not isinstance(span.get("path"), str):
            errors.append("Precedent span missing path.")
        if not isinstance(span.get("start_byte"), int):
            errors.append("Precedent span missing start_byte.")
        score = record.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        else:
            errors.append("Precedent record missing score.")
    metrics = {
        "engine_matches": len(response.primary),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "rg_partial": bool(response.debug.get("rg_partial")),
        "candidate_pool": len(context.python_files),
    }
    ok = not errors
    return CaseReport(
        case_id="precedent_search",
        query_type=request.type,
        ok=ok,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def validate_impact_slice(
    _context: ValidationContext,
    _request: QueryRequest,
    response: QueryResponse,
    prior: dict[str, QueryResponse],
) -> CaseReport:
    """Validate impact slice aggregation against prior query outputs.

    Parameters
    ----------
    _context:
        Validation context (unused).
    _request:
        Query request used for the run.
    response:
        Query response produced by the engine.
    prior:
        Prior query responses for cross-checking.

    Returns
    -------
    CaseReport
        Validation report for the impact slice case.
    """
    errors: list[str] = []
    warnings: list[str] = []
    refs = response.related.get("references", [])
    calls_in = response.related.get("calls_in", [])
    calls_out = response.related.get("calls_out", [])
    q2_response = prior.get("find_usages")
    q3_response = prior.get("call_paths")
    if q2_response is None or q3_response is None:
        warnings.append("Impact slice comparison missing prior Q2/Q3 results.")
    else:
        q2_spans = {
            key
            for record in q2_response.primary
            if isinstance(record, dict)
            if (key := _record_span_key(record)) is not None
        }
        q8_spans = {
            key
            for record in refs
            if isinstance(record, dict)
            if (key := _record_span_key(record)) is not None
        }
        if q2_spans != q8_spans:
            errors.append("Impact slice references do not match refs.find output.")

        q3_in = {
            key
            for record in q3_response.related.get("calls_in", [])
            if isinstance(record, dict)
            if (key := _call_span_key(record)) is not None
        }
        q8_in = {
            key
            for record in calls_in
            if isinstance(record, dict)
            if (key := _call_span_key(record)) is not None
        }
        if q3_in != q8_in:
            errors.append("Impact slice incoming calls do not match callgraph.slice output.")

        q3_out = {
            key
            for record in q3_response.related.get("calls_out", [])
            if isinstance(record, dict)
            if (key := _call_span_key(record)) is not None
        }
        q8_out = {
            key
            for record in calls_out
            if isinstance(record, dict)
            if (key := _call_span_key(record)) is not None
        }
        if q3_out != q8_out:
            errors.append("Impact slice outgoing calls do not match callgraph.slice output.")

    metrics = {
        "references": len(refs),
        "calls_in": len(calls_in),
        "calls_out": len(calls_out),
    }
    ok = not errors
    return CaseReport(
        case_id="impact_slice",
        query_type=_request.type,
        ok=ok,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


VALIDATORS: dict[str, CaseValidator] = {
    "symbol_resolve": validate_symbol_resolve,
    "find_usages": validate_find_usages,
    "call_paths": validate_call_paths,
    "pattern_scan": validate_pattern_scan,
    "contract_lookup": validate_contract_lookup,
    "wiring_map": validate_wiring_map,
    "precedent_search": validate_precedent_search,
    "impact_slice": validate_impact_slice,
}


def _build_validation_context(
    *,
    repo_root: Path,
    query_catalog: PackCatalog,
    wiring_catalog: PackCatalog,
    budget: QueryBudget,
    scope_paths: list[str] | None,
) -> ValidationContext:
    snippet_config = SnippetConfig(
        before_lines=budget.context_lines,
        after_lines=budget.context_lines,
    )
    search_context = SearchContext(
        repo_root=repo_root,
        query_catalog=query_catalog,
        wiring_catalog=wiring_catalog,
        snippet_config=snippet_config,
        default_budget=budget,
    )
    python_files = list_python_files(
        repo_root,
        scope_paths=scope_paths,
        max_depth=budget.max_depth,
    )
    if budget.max_files:
        python_files = python_files[: budget.max_files]
    return ValidationContext(
        repo_root=repo_root,
        query_catalog=query_catalog,
        wiring_catalog=wiring_catalog,
        search_context=search_context,
        python_files=python_files,
        budget=budget,
        scope_paths=scope_paths,
    )


def _run_cases(
    *,
    repo_root: Path,
    budget: QueryBudget,
    scope_paths: list[str] | None,
    config: dict[str, object],
    selected_cases: list[str],
) -> list[CaseReport]:
    query_root = repo_root / "docs" / "advanced_query_engine" / "query_packs"
    wiring_root = repo_root / "docs" / "advanced_query_engine" / "wiring_packs" / "packs"
    query_catalog = build_pack_catalog(query_root)
    wiring_catalog = build_pack_catalog(wiring_root)
    context = _build_validation_context(
        repo_root=repo_root,
        query_catalog=query_catalog,
        wiring_catalog=wiring_catalog,
        budget=budget,
        scope_paths=scope_paths,
    )
    service = SearchService.from_repo(
        repo_root,
        RepoServiceOptions(default_budget=budget),
    )

    reports: list[CaseReport] = []
    responses: dict[str, QueryResponse] = {}
    for case_id in selected_cases:
        definition = CASE_DEFINITIONS[case_id]
        request = _build_request(
            definition=definition,
            repo_root=repo_root,
            budget=budget,
            scope_paths=scope_paths,
            config=config,
        )
        response = service.run(request)
        responses[case_id] = response
        validator = VALIDATORS[case_id]
        reports.append(validator(context, request, response, responses))
    return reports


def _summary(reports: list[CaseReport]) -> dict[str, JSONValue]:
    total_errors = sum(len(report.errors) for report in reports)
    total_warnings = sum(len(report.warnings) for report in reports)
    return {
        "cases": len(reports),
        "errors": total_errors,
        "warnings": total_warnings,
        "ok": total_errors == 0,
    }


def _write_report(path: Path, reports: list[CaseReport], summary: dict[str, JSONValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "cases": [report.to_dict() for report in reports],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    """Run the validation harness.

    Parameters
    ----------
    argv:
        Optional command-line arguments (defaults to sys.argv[1:] when None).

    Returns
    -------
    int
        Process exit code (0 on success, non-zero on failure).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv or [])
    repo_root = args.repo_root.resolve()
    budget = QueryBudget(
        max_files=args.max_files,
        max_matches=args.max_matches,
        max_depth=args.max_depth,
        max_seconds=None,
        context_lines=args.context_lines,
    )
    scope_paths = [str(path) for path in args.scope] if args.scope else ["src", "tests"]
    config = _load_config(args.config)

    selected_cases = args.cases or list(CASE_DEFINITIONS.keys())
    unknown = [case_id for case_id in selected_cases if case_id not in CASE_DEFINITIONS]
    if unknown:
        LOGGER.error("Unknown case ids: %s", ", ".join(unknown))
        return 2

    reports = _run_cases(
        repo_root=repo_root,
        budget=budget,
        scope_paths=scope_paths,
        config=config,
        selected_cases=selected_cases,
    )
    summary = _summary(reports)
    _write_report(args.report, reports, summary)

    status = "OK" if summary["ok"] else "FAILED"
    LOGGER.info(
        "%s: %s case(s), %s error(s), %s warning(s).",
        status,
        summary["cases"],
        summary["errors"],
        summary["warnings"],
    )
    LOGGER.info("Report: %s", args.report)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
