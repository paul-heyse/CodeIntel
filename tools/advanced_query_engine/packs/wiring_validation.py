"""Wiring pack validation helpers."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path

import msgspec

from tools.advanced_query_engine.contracts import JSONValue
from tools.advanced_query_engine.packs.catalog import PackCatalog
from tools.advanced_query_engine.packs.wiring_schema import WiringPack

_MULTI_CAPTURE_RE = re.compile(r"\$\$\$([A-Za-z_][A-Za-z0-9_]*)")
_SINGLE_CAPTURE_RE = re.compile(r"(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)")
_RESERVED_CAPTURES = {"_"}
_BUILTIN_FIELDS = {"pack_id", "path", "rule_id", "enclosing_name", "enclosing_qname"}


@dataclass(frozen=True)
class PackIssue:
    """Structured wiring pack validation finding."""

    level: str
    message: str
    rule_id: str | None = None
    op: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable representation of the issue.

        Returns
        -------
        dict[str, JSONValue]
            Serialized issue payload.
        """
        payload: dict[str, JSONValue] = {"level": self.level, "message": self.message}
        if self.rule_id is not None:
            payload["rule_id"] = self.rule_id
        if self.op is not None:
            payload["op"] = self.op
        return payload


def validate_wiring_pack(pack: dict[str, JSONValue], catalog: PackCatalog) -> list[PackIssue]:
    """Validate wiring pack schemas against rule captures.

    Returns
    -------
    list[PackIssue]
        Validation findings for the pack.
    """
    issues: list[PackIssue] = []
    try:
        msgspec.convert(pack, type=WiringPack)
    except msgspec.ValidationError as exc:
        issues.append(PackIssue(level="error", message=f"Wiring pack schema error: {exc}"))
        return issues
    rule_fields, rule_ids = _collect_rule_fields(pack, catalog, issues)
    if not rule_fields:
        return issues

    postprocess_ops = pack.get("postprocess") or []
    derived_fields = _apply_postprocess(rule_fields, rule_ids, postprocess_ops, issues)
    _validate_emit(pack, derived_fields, issues)
    return issues


def resolve_pack_path(root: Path, rel_path: str) -> Path:
    """Resolve a pack file path relative to the pack root.

    Returns
    -------
    Path
        Resolved pack path.

    Raises
    ------
    ValueError
        If the resolved path escapes the pack root.
    """
    resolved = (root / rel_path).resolve()
    if not str(resolved).startswith(str(root.resolve())):
        msg = f"Path escapes pack root: {rel_path}"
        raise ValueError(msg)
    return resolved


def _collect_rule_fields(
    pack: dict[str, JSONValue],
    catalog: PackCatalog,
    issues: list[PackIssue],
) -> tuple[dict[str, set[str]], set[str]]:
    stages = pack.get("stages") or []
    rule_fields: dict[str, set[str]] = {}
    rule_ids: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("engine") != "ast_grep":
            continue
        rules_file = stage.get("rules_file")
        if not isinstance(rules_file, str):
            continue
        stage_rule_ids = {str(rule_id) for rule_id in stage.get("rule_ids") or []}
        rule_ids.update(stage_rule_ids)
        rules_path = resolve_pack_path(catalog.root, rules_file)
        if rules_path.suffix.lower() in {".yaml", ".yml"}:
            rule_pack = catalog.load_yaml(rules_path)
        else:
            rule_pack = catalog.load_json(rules_path)
        rule_fields.update(_rule_fields_from_pack(rule_pack, stage_rule_ids, issues))
    return rule_fields, rule_ids


def _rule_fields_from_pack(
    rule_pack: dict[str, JSONValue],
    rule_ids: set[str],
    issues: list[PackIssue],
) -> dict[str, set[str]]:
    rules = rule_pack.get("rules")
    if not isinstance(rules, list):
        return {}
    output: dict[str, set[str]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or rule_id not in rule_ids:
            continue
        cfg = rule.get("config")
        if not isinstance(cfg, dict):
            issues.append(
                PackIssue(level="warning", message="Rule missing config", rule_id=rule_id)
            )
            output[rule_id] = {"MATCH_NODE"}
            continue
        pattern = _rule_pattern(cfg)
        captures = _extract_captures(pattern)
        captures.add("MATCH_NODE")
        output[rule_id] = captures
    missing = sorted(rule_ids - set(output))
    if missing:
        issues.append(PackIssue(level="error", message=f"Rule ids missing from pack: {missing}"))
    return output


def _rule_pattern(cfg: dict[str, JSONValue]) -> str:
    rule_obj = cfg.get("rule")
    if isinstance(rule_obj, dict):
        pattern_value = rule_obj.get("pattern")
        if isinstance(pattern_value, str):
            return pattern_value
    return ""


def _extract_captures(pattern: str) -> set[str]:
    multi = set(_MULTI_CAPTURE_RE.findall(pattern))
    single = {
        name for name in _SINGLE_CAPTURE_RE.findall(pattern) if name not in _RESERVED_CAPTURES
    }
    return (single | multi) - _RESERVED_CAPTURES


def _apply_postprocess(
    rule_fields: dict[str, set[str]],
    rule_ids: set[str],
    ops: object,
    issues: list[PackIssue],
) -> dict[str, set[str]]:
    derived: dict[str, set[str]] = {rule_id: set(fields) for rule_id, fields in rule_fields.items()}
    for op in _coerce_ops(ops):
        op_name = str(op.get("op") or "")
        if op_name == "python.unquote_capture":
            _apply_unquote(op, derived, issues)
        elif op_name == "python.normalize_http_method":
            _apply_normalize_method(op, derived, issues)
        elif op_name == "python.upper_capture":
            _apply_upper(op, derived, issues)
        elif op_name == "python.join_argparse_subcommands":
            _apply_join_argparse(op, derived, rule_ids, issues)
    return derived


def _apply_unquote(
    op: dict[str, JSONValue],
    derived: dict[str, set[str]],
    issues: list[PackIssue],
) -> None:
    suffix = str(op.get("output_field_suffix") or "_unquoted")
    capture_names = [str(name) for name in op.get("capture_names") or []]
    for capture in capture_names:
        missing = [rule_id for rule_id, fields in derived.items() if capture not in fields]
        if missing:
            issues.append(
                PackIssue(
                    level="warning",
                    message=f"Unquote capture '{capture}' missing for rules: {missing}",
                    op="python.unquote_capture",
                )
            )
        for fields in derived.values():
            if capture in fields:
                fields.add(f"{capture}{suffix}")


def _apply_normalize_method(
    op: dict[str, JSONValue],
    derived: dict[str, set[str]],
    issues: list[PackIssue],
) -> None:
    capture = str(op.get("capture_name") or "METHOD")
    output_field = str(op.get("output_field") or "http_method")
    missing = [rule_id for rule_id, fields in derived.items() if capture not in fields]
    if missing:
        issues.append(
            PackIssue(
                level="warning",
                message=f"Normalize method capture '{capture}' missing for rules: {missing}",
                op="python.normalize_http_method",
            )
        )
    for fields in derived.values():
        if capture in fields:
            fields.add(output_field)


def _apply_upper(
    op: dict[str, JSONValue],
    derived: dict[str, set[str]],
    issues: list[PackIssue],
) -> None:
    capture = op.get("capture_name")
    if not isinstance(capture, str) or not capture:
        issues.append(
            PackIssue(
                level="warning",
                message="Upper capture missing capture_name",
                op="python.upper_capture",
            )
        )
        return
    output_field = str(op.get("output_field") or capture)
    missing = [rule_id for rule_id, fields in derived.items() if capture not in fields]
    if missing:
        issues.append(
            PackIssue(
                level="warning",
                message=f"Upper capture '{capture}' missing for rules: {missing}",
                op="python.upper_capture",
            )
        )
    for fields in derived.values():
        if capture in fields:
            fields.add(output_field)


def _apply_join_argparse(
    op: dict[str, JSONValue],
    derived: dict[str, set[str]],
    rule_ids: set[str],
    issues: list[PackIssue],
) -> None:
    add_rule = str(op.get("add_parser_rule_id") or "py.argparse.add_parser.assign")
    set_rule = str(op.get("set_defaults_rule_id") or "py.argparse.subparser.set_defaults")
    sub_var = str(op.get("subparser_var_capture") or "SUB")
    cmd_var = str(op.get("command_capture") or "CMD")
    if add_rule not in rule_ids:
        issues.append(
            PackIssue(
                level="warning",
                message=f"Argparse join missing add_parser rule '{add_rule}'",
                op="python.join_argparse_subcommands",
            )
        )
        return
    if set_rule not in rule_ids:
        issues.append(
            PackIssue(
                level="warning",
                message=f"Argparse join missing set_defaults rule '{set_rule}'",
                op="python.join_argparse_subcommands",
            )
        )
        return
    add_fields = derived.get(add_rule, set())
    set_fields = derived.get(set_rule, set())
    missing = []
    if sub_var not in add_fields:
        missing.append(f"{add_rule}.{sub_var}")
    if sub_var not in set_fields:
        missing.append(f"{set_rule}.{sub_var}")
    cmd_fields = {cmd_var, f"{cmd_var}_unquoted"}
    if not cmd_fields.intersection(add_fields):
        missing.append(f"{add_rule}.{cmd_var}")
    if missing:
        issues.append(
            PackIssue(
                level="warning",
                message=f"Argparse join missing captures: {missing}",
                op="python.join_argparse_subcommands",
            )
        )
        return
    derived[set_rule].add("ARGPARSE_CMD")


def _validate_emit(
    pack: dict[str, JSONValue],
    derived: dict[str, set[str]],
    issues: list[PackIssue],
) -> None:
    emit = pack.get("emit") or {}
    entry_key_template = emit.get("entry_key_template")
    entry_key_by_rule = emit.get("entry_key_by_rule") or {}
    hook_span_by_rule = emit.get("hook_span_by_rule") or {}
    target_hint_by_rule = emit.get("target_symbol_hint_by_rule") or {}
    hook_span_capture = emit.get("hook_span_capture")
    target_hint_capture = emit.get("target_symbol_hint_capture")

    for rule_id, fields in derived.items():
        template = entry_key_by_rule.get(rule_id) or entry_key_template
        if isinstance(template, str):
            missing = _missing_template_fields(template, fields)
            if missing:
                issues.append(
                    PackIssue(
                        level="error",
                        message=f"Entry template missing fields: {missing}",
                        rule_id=rule_id,
                    )
                )
        elif template is not None:
            issues.append(
                PackIssue(
                    level="warning",
                    message="Entry key template is not a string",
                    rule_id=rule_id,
                )
            )

        hook_capture = hook_span_by_rule.get(rule_id) or hook_span_capture
        if isinstance(hook_capture, str) and hook_capture not in fields:
            issues.append(
                PackIssue(
                    level="warning",
                    message=f"Hook span capture '{hook_capture}' missing",
                    rule_id=rule_id,
                )
            )
        target_capture = target_hint_by_rule.get(rule_id) or target_hint_capture
        if isinstance(target_capture, str) and target_capture not in fields:
            issues.append(
                PackIssue(
                    level="warning",
                    message=f"Target hint capture '{target_capture}' missing",
                    rule_id=rule_id,
                )
            )


def _missing_template_fields(template: str, fields: set[str]) -> list[str]:
    required = _template_fields(template)
    available = fields | _BUILTIN_FIELDS
    return sorted(required - available)


def _template_fields(template: str) -> set[str]:
    formatter = string.Formatter()
    fields: set[str] = set()
    for _, field_name, _, _ in formatter.parse(template):
        if not field_name:
            continue
        fields.add(_base_field(field_name))
    return fields


def _base_field(field_name: str) -> str:
    for sep in (".", "["):
        if sep in field_name:
            return field_name.split(sep, 1)[0]
    return field_name


def _coerce_ops(value: object) -> list[dict[str, JSONValue]]:
    if not value:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


__all__ = ["PackIssue", "resolve_pack_path", "validate_wiring_pack"]
