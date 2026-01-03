"""Wiring map handler (Q6)."""

from __future__ import annotations

import ast
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tools.advanced_query_engine.backends.astgrep_backend import run_rules_on_root, select_rules
from tools.advanced_query_engine.backends.rpygrep_backend import (
    RpygrepQuery,
    RpygrepResult,
    run_pattern_group,
)
from tools.advanced_query_engine.context import SearchContext
from tools.advanced_query_engine.contracts import QueryBudget, QueryRequest, QueryResponse, Span
from tools.advanced_query_engine.handlers.common import load_rpygrep_preset
from tools.advanced_query_engine.packs.wiring_validation import (
    resolve_pack_path,
    validate_wiring_pack,
)
from tools.advanced_query_engine.util.hashing import stable_hex_digest
from tools.advanced_query_engine.util.snippets import SnippetRequest, build_snippet
from tools.advanced_query_engine.util.template import safe_format

if TYPE_CHECKING:
    from tools.advanced_query_engine.backends.astgrep_backend import AstGrepMatch
    from tools.advanced_query_engine.packs.wiring_validation import PackIssue


class WiringPackError(RuntimeError):
    """Error raised when wiring pack configuration is invalid."""


class WiringPackMissingStagesError(WiringPackError):
    """Wiring pack has no stages."""

    def __init__(self) -> None:
        super().__init__("Wiring pack has no stages.")


class WiringPackInvalidStageError(WiringPackError):
    """Wiring pack has an invalid first stage."""

    def __init__(self) -> None:
        super().__init__("Wiring pack must start with rpygrep stage.")


class WiringPackTypeError(TypeError):
    """Wiring pack config field has an invalid type."""

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name
        super().__init__(f"{field_name} must be a string.")


@dataclass
class _WiringRecord:
    path: str
    rule_id: str
    match_span: Span
    captures: dict[str, str | list[str]]
    capture_spans: dict[str, Span | list[Span]]
    rg_pattern_ids: list[str]
    enclosing_def: dict[str, object] | None


@dataclass(frozen=True)
class _WiringExecution:
    context: SearchContext
    budget: QueryBudget
    allow_cross_file: bool
    scope_paths: list[str] | None
    deadline: float | None


def _deadline(budget: QueryBudget) -> float | None:
    if budget.max_seconds is None:
        return None
    return time.monotonic() + float(budget.max_seconds)


def _budget_exhausted(budget: QueryBudget, deadline: float | None, count: int) -> bool:
    return (budget.max_matches > 0 and count >= budget.max_matches) or (
        deadline is not None and time.monotonic() >= deadline
    )


def _unquote_literal(value: str) -> str | None:
    text = value.strip()
    if not text or text[0] not in {'"', "'"}:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, str) else None


def _normalize_http_method(value: str | None) -> str:
    if not value:
        return "*"
    return value.strip().upper()


def _simple_name(expr: str) -> str | None:
    expr = expr.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
        return expr
    return None


def _capture_text(value: str | list[str] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return ", ".join(value)
    return value


def _record_from_match(
    match: AstGrepMatch, context: SearchContext, rg_ids: list[str]
) -> _WiringRecord:
    line_index = context.line_index(match.path)
    match_span = Span(
        path=match.path,
        start_byte=match.match_start,
        end_byte=match.match_end,
        **line_index.span_to_range(match.match_start, match.match_end),
    )
    captures: dict[str, str | list[str]] = {}
    capture_spans: dict[str, Span | list[Span]] = {}
    for name, values in match.captures.items():
        if len(values) == 1:
            captures[name] = values[0].text
            capture_spans[name] = Span(
                path=match.path,
                start_byte=values[0].start_byte,
                end_byte=values[0].end_byte,
                **line_index.span_to_range(values[0].start_byte, values[0].end_byte),
            )
        else:
            captures[name] = [value.text for value in values]
            capture_spans[name] = [
                Span(
                    path=match.path,
                    start_byte=value.start_byte,
                    end_byte=value.end_byte,
                    **line_index.span_to_range(value.start_byte, value.end_byte),
                )
                for value in values
            ]

    enclosing = None
    try:
        def_index = context.def_index(match.path)
        enc = def_index.enclosing_def(match.match_start)
        if enc is not None:
            enclosing = {"name": enc.name, "qname": enc.qname, "kind": enc.kind}
    except (FileNotFoundError, ValueError):
        enclosing = None

    return _WiringRecord(
        path=match.path,
        rule_id=match.rule_id,
        match_span=match_span,
        captures=captures,
        capture_spans=capture_spans,
        rg_pattern_ids=rg_ids,
        enclosing_def=enclosing,
    )


def _apply_unquote_capture(records: list[_WiringRecord], op: dict[str, object]) -> None:
    suffix = op.get("output_field_suffix") or "_unquoted"
    capture_names = op.get("capture_names") or []
    for record in records:
        for cap_name in capture_names:
            value = _capture_text(record.captures.get(cap_name))
            if value is None:
                continue
            unquoted = _unquote_literal(value)
            if unquoted is not None:
                record.captures[f"{cap_name}{suffix}"] = unquoted


def _apply_normalize_http_method(records: list[_WiringRecord], op: dict[str, object]) -> None:
    cap_name = op.get("capture_name") or "METHOD"
    out_field = op.get("output_field") or "http_method"
    for record in records:
        method = _capture_text(record.captures.get(cap_name))
        record.captures[out_field] = _normalize_http_method(method)


def _apply_upper_capture(records: list[_WiringRecord], op: dict[str, object]) -> None:
    cap_name = op.get("capture_name")
    if not cap_name:
        return
    out_field = op.get("output_field") or cap_name
    for record in records:
        value = _capture_text(record.captures.get(cap_name))
        if value is not None:
            record.captures[out_field] = value.upper()


_KNOWN_HTTP_METHODS = {
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "OPTIONS",
    "HEAD",
    "TRACE",
    "CONNECT",
    "WEBSOCKET",
}
_METHODS_ARG_RE = re.compile(r"methods?\\s*=\\s*(?P<value>\\[[^\\]]*\\]|\\([^)]*\\)|\\{[^}]*\\})")


def _methods_from_args(args: list[str] | None) -> list[str] | None:
    if not args:
        return None
    for arg in args:
        match = _METHODS_ARG_RE.search(arg)
        if not match:
            continue
        raw = match.group("value")
        methods = _coerce_methods(raw)
        if methods:
            return methods
    return None


def _coerce_methods(raw: str) -> list[str] | None:
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, str):
        return [_normalize_http_method(parsed)]
    if isinstance(parsed, (list, tuple, set)):
        normalized = [_normalize_http_method(item) for item in parsed if isinstance(item, str)]
        return [item for item in normalized if item]
    tokens = [token.upper() for token in re.findall(r"\\b[A-Za-z]+\\b", raw)]
    matches = [token for token in tokens if token in _KNOWN_HTTP_METHODS]
    return matches or None


def _default_from_args(args: list[str] | None) -> str | None:
    if not args:
        return None
    candidate = args[0]
    if "=" in candidate:
        candidate = candidate.split("=", 1)[1]
    unquoted = _unquote_literal(candidate)
    if unquoted is not None:
        return unquoted
    return candidate.strip() if candidate.strip() else None


def _args_list(record: _WiringRecord) -> list[str] | None:
    args_value = record.captures.get("ARGS")
    if isinstance(args_value, list):
        return args_value
    if isinstance(args_value, str):
        return [args_value]
    return None


def _set_if(config: dict[str, object], key: str, value: str | None) -> None:
    if value:
        config[key] = value


def _base_config(record: _WiringRecord, args_list: list[str] | None) -> dict[str, object]:
    config: dict[str, object] = {}
    captures = record.captures
    _set_if(config, "path", _capture_text(captures.get("PATH_unquoted") or captures.get("PATH")))
    method_value = _capture_text(captures.get("http_method") or captures.get("METHOD"))
    if method_value:
        config["http_method"] = _normalize_http_method(method_value)
    methods = _methods_from_args(args_list)
    if methods:
        config["methods"] = methods
    _set_if(config, "group", _capture_text(captures.get("GROUP_unquoted") or captures.get("GROUP")))
    command_value = _capture_text(
        captures.get("ARGPARSE_CMD") or captures.get("CMD_unquoted") or captures.get("CMD")
    )
    _set_if(config, "command", command_value)
    _set_if(config, "handler_hint", _capture_text(captures.get("HANDLER")))
    return config


def _env_config(record: _WiringRecord, args_list: list[str] | None) -> dict[str, object]:
    config: dict[str, object] = {}
    captures = record.captures
    _set_if(config, "key", _capture_text(captures.get("KEY_unquoted") or captures.get("KEY")))
    default_value = _default_from_args(args_list)
    if default_value is not None:
        config["default"] = default_value
    return config


def _config_from_record(pack_id: str, record: _WiringRecord) -> dict[str, object] | None:
    args_list = _args_list(record)
    config = _base_config(record, args_list)
    if "env" in pack_id:
        config.update(_env_config(record, args_list))
    return config or None


_POSTPROCESS_HANDLERS: dict[str, Callable[[list[_WiringRecord], dict[str, object]], None]] = {
    "python.unquote_capture": _apply_unquote_capture,
    "python.normalize_http_method": _apply_normalize_http_method,
    "python.upper_capture": _apply_upper_capture,
}


def _apply_join_argparse_subcommands(records: list[_WiringRecord], op: dict[str, object]) -> None:
    add_rule = op.get("add_parser_rule_id", "py.argparse.add_parser.assign")
    set_rule = op.get("set_defaults_rule_id", "py.argparse.subparser.set_defaults")
    sub_var = op.get("subparser_var_capture", "SUB")
    cmd_var = op.get("command_capture", "CMD")
    by_file: dict[str, dict[str, str]] = {}
    for record in records:
        if record.rule_id != add_rule:
            continue
        sub_name = _capture_text(record.captures.get(sub_var))
        cmd = _capture_text(record.captures.get(f"{cmd_var}_unquoted")) or _capture_text(
            record.captures.get(cmd_var)
        )
        if sub_name and cmd:
            by_file.setdefault(record.path, {})[sub_name] = cmd
    for record in records:
        if record.rule_id != set_rule:
            continue
        sub_name = _capture_text(record.captures.get(sub_var))
        if not sub_name:
            continue
        by_record = by_file.get(record.path)
        if by_record and sub_name in by_record:
            record.captures["ARGPARSE_CMD"] = by_record[sub_name]


def _postprocess_records(
    pack: dict[str, object], records: list[_WiringRecord]
) -> list[_WiringRecord]:
    ops = pack.get("postprocess") or []
    for op in ops:
        handler = _POSTPROCESS_HANDLERS.get(op.get("op"))
        if handler is not None:
            handler(records, op)
    for op in ops:
        if op.get("op") == "python.join_argparse_subcommands":
            _apply_join_argparse_subcommands(records, op)
            break
    return records


def _resolve_handler_target(
    context: SearchContext,
    record: _WiringRecord,
    handler_expr: str,
    *,
    allow_cross_file: bool,
    preset_id: str,
) -> dict[str, object] | None:
    name = _simple_name(handler_expr)
    if not name:
        return None
    try:
        local_index = context.def_index(record.path)
    except (FileNotFoundError, ValueError):
        local_index = None
    if local_index is not None:
        local_defs = local_index.by_name(name)
        if local_defs:
            def_rec = local_defs[0]
            return {"name": def_rec.name, "qname": def_rec.qname, "kind": def_rec.kind}
    if not allow_cross_file:
        return None

    preset = load_rpygrep_preset(context.query_catalog, preset_id)
    pattern_group = {
        "pattern_group_id": f"rg.resolve.{name}",
        "patterns": [
            {"pattern": rf"\bdef\s+{re.escape(name)}\b", "is_regex": True, "priority": 10}
        ],
        "globs": ["**/*.py"],
    }
    budget = context.default_budget
    candidates = run_pattern_group(
        RpygrepQuery(
            repo_root=context.repo_root,
            preset=preset,
            pattern_group=pattern_group,
            budget=budget,
            scope_paths=None,
            cache=context.cache,
        )
    )
    for rel_path in candidates.files_to_patterns:
        try:
            def_index = context.def_index(rel_path)
            defs = def_index.by_name(name)
        except (FileNotFoundError, ValueError):
            continue
        if defs:
            def_rec = defs[0]
            return {"name": def_rec.name, "qname": def_rec.qname, "kind": def_rec.kind}
    return None


def _resolve_hook_span(
    record: _WiringRecord, hook_span_by_rule: dict[str, object], emit: dict[str, object]
) -> Span:
    hook_capture = hook_span_by_rule.get(record.rule_id) or emit.get("hook_span_capture")
    hook_value = record.capture_spans.get(str(hook_capture)) if hook_capture else None
    if isinstance(hook_value, list):
        return hook_value[0] if hook_value else record.match_span
    if isinstance(hook_value, Span):
        return hook_value
    return record.match_span


def _resolve_target(
    context: SearchContext,
    record: _WiringRecord,
    target_hint_by_rule: dict[str, object],
    emit: dict[str, object],
    *,
    allow_cross_file: bool,
) -> dict[str, object] | None:
    handler_capture = target_hint_by_rule.get(record.rule_id) or emit.get(
        "target_symbol_hint_capture"
    )
    handler_expr = (
        _capture_text(record.captures.get(str(handler_capture))) if handler_capture else None
    )
    if not handler_expr:
        return record.enclosing_def
    resolved = _resolve_handler_target(
        context,
        record,
        handler_expr,
        allow_cross_file=allow_cross_file,
        preset_id="rg.audit_deterministic",
    )
    return resolved if resolved is not None else record.enclosing_def


def _entry_key_for_record(
    record: _WiringRecord,
    pack_id: str,
    entry_key_by_rule: dict[str, object],
    entry_key_template: str,
) -> str:
    values: dict[str, object] = {
        **record.captures,
        "pack_id": pack_id,
        "path": record.path,
        "rule_id": record.rule_id,
    }
    if record.enclosing_def:
        values.setdefault("enclosing_name", record.enclosing_def.get("name"))
        values.setdefault("enclosing_qname", record.enclosing_def.get("qname"))
    entry_template = entry_key_by_rule.get(record.rule_id) or entry_key_template
    return safe_format(str(entry_template), values)


@dataclass(frozen=True)
class _EdgeContext:
    pack: dict[str, object]
    pack_id: str
    entry_kind: str
    entry_key: str
    hook: Span
    target: dict[str, object] | None
    config: dict[str, object] | None


def _build_edge(
    record: _WiringRecord,
    context: SearchContext,
    edge: _EdgeContext,
) -> dict[str, object]:
    evidence = build_snippet(
        SnippetRequest(
            source=context.cache.read_bytes(record.path),
            span=edge.hook,
            config=context.snippet_config,
            line_index=context.line_index(record.path),
        )
    )
    edge_id = stable_hex_digest(
        [
            str(edge.pack_id),
            str(edge.entry_kind),
            str(edge.entry_key),
            edge.hook.path,
            str(edge.hook.start_byte),
            str(edge.hook.end_byte),
            str((edge.target or {}).get("qname") if isinstance(edge.target, dict) else ""),
        ]
    )
    payload = {
        "edge_id": edge_id,
        "pack_id": edge.pack_id,
        "framework": edge.pack.get("framework"),
        "entry_kind": edge.entry_kind,
        "entry_key": edge.entry_key,
        "hook_span": edge.hook.to_dict(),
        "target": edge.target,
        "match": {
            "path": record.path,
            "rule_id": record.rule_id,
            "match_span": record.match_span.to_dict(),
            "captures": record.captures,
            "rg_pattern_ids": record.rg_pattern_ids,
        },
        "evidence": evidence.to_dict(),
    }
    if edge.config:
        payload["config"] = edge.config
    return payload


def _emit_edges(
    pack: dict[str, object],
    context: SearchContext,
    records: list[_WiringRecord],
    *,
    allow_cross_file: bool,
) -> list[dict[str, object]]:
    emit = pack.get("emit") or {}
    entry_kind = pack.get("entry_kind") or pack.get("entryKind") or "wiring"
    pack_id = pack.get("pack_id") or "<unknown>"
    entry_key_template = str(emit.get("entry_key_template") or "{pack_id}:{path}:{rule_id}")
    entry_key_by_rule = emit.get("entry_key_by_rule") or {}
    target_hint_by_rule = emit.get("target_symbol_hint_by_rule") or {}
    hook_span_by_rule = emit.get("hook_span_by_rule") or {}

    edges: list[dict[str, object]] = []
    for record in records:
        hook = _resolve_hook_span(record, hook_span_by_rule, emit)
        target = _resolve_target(
            context,
            record,
            target_hint_by_rule,
            emit,
            allow_cross_file=allow_cross_file,
        )
        config = _config_from_record(pack_id, record)
        entry_key = _entry_key_for_record(record, pack_id, entry_key_by_rule, entry_key_template)
        edge = _EdgeContext(
            pack=pack,
            pack_id=pack_id,
            entry_kind=entry_kind,
            entry_key=entry_key,
            hook=hook,
            target=target,
            config=config,
        )
        edges.append(_build_edge(record, context, edge))

    edges.sort(key=lambda item: (item["entry_kind"], item["entry_key"], item["hook_span"]["path"]))
    return edges


def _validation_payload(issues: list[PackIssue]) -> dict[str, list[dict[str, object]]]:
    errors = [issue.to_dict() for issue in issues if issue.level == "error"]
    warnings = [issue.to_dict() for issue in issues if issue.level != "error"]
    return {"errors": errors, "warnings": warnings}


def _require_stages(pack: dict[str, object]) -> list[object]:
    stages = pack.get("stages") or []
    if not stages:
        raise WiringPackMissingStagesError
    return list(stages)


def _require_rpygrep_stage(stages: list[object]) -> dict[str, object]:
    first = stages[0]
    if not isinstance(first, dict) or first.get("engine") != "rpygrep":
        raise WiringPackInvalidStageError
    return first


def _load_pattern_group(context: SearchContext, stage: dict[str, object]) -> dict[str, object]:
    pattern_group_file = stage.get("pattern_group_file")
    if not isinstance(pattern_group_file, str):
        field_name = "pattern_group_file"
        raise WiringPackTypeError(field_name)
    pattern_group_path = resolve_pack_path(context.wiring_catalog.root, pattern_group_file)
    return context.wiring_catalog.load_json(pattern_group_path)


def _load_rule_pack(context: SearchContext, rules_path: Path) -> dict[str, object]:
    if rules_path.suffix.lower() in {".yaml", ".yml"}:
        return context.wiring_catalog.load_yaml(rules_path)
    return context.wiring_catalog.load_json(rules_path)


def _ast_grep_records_for_stage(
    context: SearchContext,
    stage: dict[str, object],
    rg_result: RpygrepResult,
    budget: QueryBudget,
    deadline: float | None,
) -> tuple[list[_WiringRecord], bool]:
    if stage.get("engine") != "ast_grep":
        return [], False
    rules_file = stage.get("rules_file")
    if not isinstance(rules_file, str):
        return [], False
    rules_path = resolve_pack_path(context.wiring_catalog.root, rules_file)
    rule_pack = _load_rule_pack(context, rules_path)
    rule_ids = [str(rule_id) for rule_id in stage.get("rule_ids") or []]
    rules = select_rules(rule_pack, rule_ids)

    records: list[_WiringRecord] = []
    partial = False
    for rel_path in sorted(rg_result.files_to_patterns.keys()):
        if _budget_exhausted(budget, deadline, len(records)):
            partial = True
            break
        if not rel_path.endswith(".py"):
            continue
        try:
            root = context.ast_grep_root(rel_path)
        except (FileNotFoundError, ValueError):
            continue
        matches = run_rules_on_root(rel_path=rel_path, root=root, rules=rules)
        pattern_ids = rg_result.files_to_patterns.get(rel_path, [])
        for match in matches:
            records.append(_record_from_match(match, context, pattern_ids))
            if _budget_exhausted(budget, deadline, len(records)):
                partial = True
                break
        if partial:
            break
    return records, partial


def _execute_pack(
    *,
    pack: dict[str, object],
    exec_ctx: _WiringExecution,
) -> dict[str, object]:
    issues = validate_wiring_pack(pack, exec_ctx.context.wiring_catalog)
    validation = _validation_payload(issues)
    if validation["errors"]:
        return {
            "pack_id": pack.get("pack_id"),
            "entry_kind": pack.get("entry_kind"),
            "framework": pack.get("framework"),
            "edges": [],
            "partial": True,
            "skipped": True,
            "validation": validation,
        }

    stages = _require_stages(pack)
    first = _require_rpygrep_stage(stages)

    preset_id = str(first.get("preset") or "rg.default_interactive")
    preset = load_rpygrep_preset(exec_ctx.context.wiring_catalog, preset_id)
    pattern_group = _load_pattern_group(exec_ctx.context, first)

    rg_result = run_pattern_group(
        RpygrepQuery(
            repo_root=exec_ctx.context.repo_root,
            preset=preset,
            pattern_group=pattern_group,
            budget=exec_ctx.budget,
            scope_paths=exec_ctx.scope_paths,
            cache=exec_ctx.context.cache,
        )
    )

    records: list[_WiringRecord] = []
    partial = rg_result.partial
    for stage in stages[1:]:
        if not isinstance(stage, dict):
            continue
        stage_records, stage_partial = _ast_grep_records_for_stage(
            exec_ctx.context,
            stage,
            rg_result,
            exec_ctx.budget,
            exec_ctx.deadline,
        )
        records.extend(stage_records)
        if stage_partial:
            partial = True
            break

    records = _postprocess_records(pack, records)
    edges = _emit_edges(pack, exec_ctx.context, records, allow_cross_file=exec_ctx.allow_cross_file)

    return {
        "pack_id": pack.get("pack_id"),
        "entry_kind": pack.get("entry_kind"),
        "framework": pack.get("framework"),
        "edges": edges,
        "partial": partial,
        "validation": validation,
        "debug": {
            "rg_files": sorted(rg_result.files_to_patterns.keys()),
            "rg_partial": rg_result.partial,
        },
    }


def _resolve_pack_paths(pack_ids: object, available: dict[str, Path]) -> list[Path]:
    if pack_ids is None:
        return list(available.values())
    if isinstance(pack_ids, list):
        pack_id_list = [str(item) for item in pack_ids]
    elif isinstance(pack_ids, str):
        pack_id_list = [pack_ids]
    else:
        msg = "pack_ids must be a string or list of strings."
        raise TypeError(msg)
    return [available[pid] for pid in pack_id_list if pid in available]


def _run_packs(
    pack_list: list[Path], exec_ctx: _WiringExecution
) -> tuple[list[dict[str, object]], bool]:
    results: list[dict[str, object]] = []
    budget_exhausted = False
    for pack_path in pack_list:
        if _budget_exhausted(exec_ctx.budget, exec_ctx.deadline, 0):
            budget_exhausted = True
            break
        pack = exec_ctx.context.wiring_catalog.load_json(pack_path)
        results.append(_execute_pack(pack=pack, exec_ctx=exec_ctx))
    return results, budget_exhausted


def _edges_from_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for result in results:
        edges.extend(result.get("edges") or [])
    edges.sort(key=lambda item: (item.get("entry_kind"), item.get("entry_key")))
    return edges


def _validation_summary(results: list[dict[str, object]]) -> dict[str, int]:
    errors = 0
    warnings = 0
    skipped = 0
    partial = 0
    for result in results:
        validation = result.get("validation")
        if isinstance(validation, dict):
            errors += len(validation.get("errors") or [])
            warnings += len(validation.get("warnings") or [])
        if result.get("skipped"):
            skipped += 1
        if result.get("partial"):
            partial += 1
    return {
        "errors": errors,
        "warnings": warnings,
        "skipped": skipped,
        "partial": partial,
    }


def handle(request: QueryRequest, context: SearchContext) -> QueryResponse:
    """Execute wiring packs and emit wiring edges.

    Parameters
    ----------
    request:
        Query request containing wiring options.
    context:
        Search context providing indices and catalogs.

    Returns
    -------
    QueryResponse
        Query response with wiring edges and per-pack details.

    """
    budget = request.budget or context.default_budget
    if not isinstance(budget, QueryBudget):
        budget = QueryBudget()

    options = request.options or {}
    allow_cross_file = bool(options.get("allow_cross_file_resolution", True))
    exec_ctx = _WiringExecution(
        context=context,
        budget=budget,
        allow_cross_file=allow_cross_file,
        scope_paths=request.scope_paths,
        deadline=_deadline(budget),
    )
    pack_list = _resolve_pack_paths(options.get("pack_ids"), context.wiring_catalog.wiring_packs)
    results, budget_exhausted = _run_packs(pack_list, exec_ctx)
    edges = _edges_from_results(results)
    validation_summary = _validation_summary(results)

    summary = f"Emitted {len(edges)} wiring edges from {len(results)} pack(s)."
    related = {"by_pack": results}
    debug = {
        "pack_count": len(results),
        "budget_exhausted": budget_exhausted,
        "validation": validation_summary,
    }
    return QueryResponse(summary=summary, primary=edges, related=related, debug=debug)


__all__ = ["handle"]
