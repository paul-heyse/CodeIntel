"""Ast-grep backend for structural pattern matching."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast

try:
    from ast_grep_py import SgRoot
except ModuleNotFoundError as exc:  # pragma: no cover
    SgRoot = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class AstGrepCapture:
    """Single capture value from ast-grep."""

    name: str
    text: str
    start_byte: int
    end_byte: int
    start_line0: int
    start_col0: int
    end_line0: int
    end_col0: int


@dataclass(frozen=True)
class AstGrepMatch:
    """Match record emitted from ast-grep."""

    path: str
    rule_id: str
    match_start: int
    match_end: int
    match_kind: str
    match_text: str
    captures: dict[str, list[AstGrepCapture]]


_SINGLE_RE = re.compile(r"(?<!\$)\$(\w+)")
_MULTI_RE = re.compile(r"\$\$\$(\w+)")
_RESERVED = {"_"}


class AstGrepNode(Protocol):
    """Protocol for AST-grep node operations used by this backend."""

    def find_all(self, cfg: dict[str, object]) -> Iterable[object]: ...


class AstGrepRoot(Protocol):
    """Protocol for AST-grep root wrapper objects."""

    def root(self) -> AstGrepNode:
        """Return the root node for traversal."""
        ...


def _extract_capture_names(pattern: str) -> tuple[list[str], list[str]]:
    multi = list(dict.fromkeys(_MULTI_RE.findall(pattern)))
    single = [name for name in _SINGLE_RE.findall(pattern) if name not in _RESERVED]
    single = [name for name in single if name not in multi]
    single = list(dict.fromkeys(single))
    return single, multi


def _node_span(node: object) -> tuple[int, int, int, int, int, int]:
    r = node.range()
    return (
        int(r.start.index),
        int(r.end.index),
        int(r.start.line),
        int(r.start.column),
        int(r.end.line),
        int(r.end.column),
    )


def _load_ast_grep() -> type:
    if SgRoot is None:
        msg = f"ast-grep-py is not importable. Import error: {_IMPORT_ERROR}"
        raise RuntimeError(msg)
    return SgRoot


def parse_ast_grep_source(source: str) -> AstGrepRoot:
    """Parse source code into an ast-grep root object.

    Returns
    -------
    AstGrepRoot
        Parsed ast-grep root wrapper.
    """
    sg_root = _load_ast_grep()
    return cast("AstGrepRoot", sg_root(source, "python"))


def _iter_rules(
    rules: Iterable[dict[str, object]],
) -> Iterable[tuple[str, dict[str, object], list[str], list[str]]]:
    for rule in rules:
        rule_id = str(rule.get("rule_id"))
        cfg = rule.get("config")
        if not isinstance(cfg, dict):
            continue
        pattern = _rule_pattern(cfg)
        single_caps, multi_caps = _extract_capture_names(pattern) if pattern else ([], [])
        yield rule_id, cfg, single_caps, multi_caps


def _rule_pattern(cfg: dict[str, object]) -> str:
    rule_obj = cfg.get("rule")
    if isinstance(rule_obj, dict):
        pattern_value = rule_obj.get("pattern")
        if isinstance(pattern_value, str):
            return pattern_value
    return ""


def _collect_captures(
    match: object,
    single_caps: list[str],
    multi_caps: list[str],
) -> dict[str, list[AstGrepCapture]]:
    captures: dict[str, list[AstGrepCapture]] = {}

    for name in single_caps:
        captured = match.get_match(name)
        if captured is None:
            continue
        captures.setdefault(name, []).append(_capture_from_node(name, captured))

    for name in multi_caps:
        for captured in match.get_multiple_matches(name) or []:
            captures.setdefault(name, []).append(_capture_from_node(name, captured))

    captures.setdefault("MATCH_NODE", []).append(_capture_from_node("MATCH_NODE", match))
    return captures


def _capture_from_node(name: str, node: object) -> AstGrepCapture:
    cs, ce, csl0, csc0, cel0, cec0 = _node_span(node)
    return AstGrepCapture(
        name=name,
        text=node.text(),
        start_byte=cs,
        end_byte=ce,
        start_line0=csl0,
        start_col0=csc0,
        end_line0=cel0,
        end_col0=cec0,
    )


def run_rules_on_file(
    *,
    rel_path: str,
    source: str,
    rules: Iterable[dict[str, object]],
) -> list[AstGrepMatch]:
    """Execute ast-grep rules against a source string.

    Returns
    -------
    list[AstGrepMatch]
        Sorted match records.
    """
    root = parse_ast_grep_source(source)
    return run_rules_on_root(rel_path=rel_path, root=root, rules=rules)


def run_rules_on_root(
    *,
    rel_path: str,
    root: AstGrepRoot,
    rules: Iterable[dict[str, object]],
) -> list[AstGrepMatch]:
    """Execute ast-grep rules against a parsed root.

    Returns
    -------
    list[AstGrepMatch]
        Sorted match records.
    """
    node = root.root()

    results: list[AstGrepMatch] = []
    for rule_id, cfg, single_caps, multi_caps in _iter_rules(rules):
        for match in node.find_all(cfg):
            captures = _collect_captures(match, single_caps, multi_caps)
            ms, me, _, _, _, _ = _node_span(match)
            results.append(
                AstGrepMatch(
                    path=rel_path,
                    rule_id=rule_id,
                    match_start=ms,
                    match_end=me,
                    match_kind=match.kind(),
                    match_text=match.text(),
                    captures=captures,
                )
            )

    results.sort(key=lambda item: (item.path, item.match_start, item.match_end, item.rule_id))
    return results


def select_rules(rule_pack: dict[str, object], rule_ids: Iterable[str]) -> list[dict[str, object]]:
    """Select rule configs from a rule pack by id.

    Returns
    -------
    list[dict[str, object]]
        Rule definitions matching the requested ids.

    Raises
    ------
    ValueError
        If requested rule ids are not present in the pack.
    """
    all_rules = rule_pack.get("rules") or []
    wanted = set(rule_ids)
    selected = [rule for rule in all_rules if rule.get("rule_id") in wanted]
    missing = sorted(wanted - {rule.get("rule_id") for rule in selected})
    if missing:
        msg = f"Rule ids not found in rule pack: {missing}"
        raise ValueError(msg)
    output: list[dict[str, object]] = []
    for rule in selected:
        if "config" not in rule:
            msg = f"Rule {rule.get('rule_id')} missing config"
            raise ValueError(msg)
        output.append({"rule_id": rule["rule_id"], "config": rule["config"]})
    return output


__all__ = [
    "AstGrepCapture",
    "AstGrepMatch",
    "AstGrepRoot",
    "parse_ast_grep_source",
    "run_rules_on_file",
    "run_rules_on_root",
    "select_rules",
]
