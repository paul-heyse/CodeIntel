from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

try:
    from ast_grep_py import SgRoot  # type: ignore
except Exception as e:  # pragma: no cover
    SgRoot = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class CaptureValue:
    name: str
    text: str
    start_byte: int
    end_byte: int
    start_line0: int
    start_col0: int
    end_line0: int
    end_col0: int


@dataclass(frozen=True)
class AGMatchRecord:
    path: str
    rule_id: str
    match_start: int
    match_end: int
    match_kind: str
    match_text: str
    captures: Dict[str, List[CaptureValue]]  # name -> list (multi or single, normalized to list)


_SINGLE_RE = re.compile(r"(?<!\$)\$(\w+)")
_MULTI_RE = re.compile(r"\$\$\$(\w+)")
# Keywords that appear in patterns but are not metavars
_RESERVED = {"_"}


def _extract_capture_names(pattern: str) -> Tuple[List[str], List[str]]:
    multi = list(dict.fromkeys(_MULTI_RE.findall(pattern)))
    single = list(dict.fromkeys([s for s in _SINGLE_RE.findall(pattern) if s not in _RESERVED]))
    # Remove any single that appears as multi name (rare but safe)
    single = [s for s in single if s not in multi]
    return single, multi


def _node_span(node: Any) -> Tuple[int, int, int, int, int, int]:
    r = node.range()
    # ast-grep positions are 0-indexed line/col; index is absolute offset
    return (
        int(r.start.index), int(r.end.index),
        int(r.start.line), int(r.start.column),
        int(r.end.line), int(r.end.column),
    )


def run_ast_grep_rules_on_file(
    *,
    rel_path: str,
    src_text: str,
    rules: List[dict],
) -> List[AGMatchRecord]:
    if SgRoot is None:
        raise RuntimeError(f"ast-grep-py is not importable. Install ast-grep-py. Import error: {_IMPORT_ERROR}")

    root = SgRoot(src_text, "python")
    node = root.root()

    out: List[AGMatchRecord] = []

    for rule in rules:
        rule_id = rule["rule_id"]
        cfg = rule["config"]
        pattern = ((cfg.get("rule") or {}).get("pattern") or "")
        single_caps, multi_caps = _extract_capture_names(pattern)

        # Run config-based find_all for constraints/utils support.
        matches = node.find_all(cfg)  # type: ignore[call-arg]

        for m in matches:
            ms, me, sl0, sc0, el0, ec0 = _node_span(m)
            caps: Dict[str, List[CaptureValue]] = {}

            for name in single_caps:
                c = m.get_match(name)
                if c is None:
                    continue
                cs, ce, csl0, csc0, cel0, cec0 = _node_span(c)
                caps.setdefault(name, []).append(CaptureValue(
                    name=name,
                    text=c.text(),
                    start_byte=cs,
                    end_byte=ce,
                    start_line0=csl0,
                    start_col0=csc0,
                    end_line0=cel0,
                    end_col0=cec0,
                ))

            for name in multi_caps:
                cs = m.get_multiple_matches(name)
                if not cs:
                    continue
                for c in cs:
                    csb, ceb, csl0, csc0, cel0, cec0 = _node_span(c)
                    caps.setdefault(name, []).append(CaptureValue(
                        name=name,
                        text=c.text(),
                        start_byte=csb,
                        end_byte=ceb,
                        start_line0=csl0,
                        start_col0=csc0,
                        end_line0=cel0,
                        end_col0=cec0,
                    ))

            # Always include MATCH_NODE span for emit/hook anchoring.
            caps.setdefault("MATCH_NODE", []).append(CaptureValue(
                name="MATCH_NODE",
                text=m.text(),
                start_byte=ms,
                end_byte=me,
                start_line0=sl0,
                start_col0=sc0,
                end_line0=el0,
                end_col0=ec0,
            ))

            out.append(AGMatchRecord(
                path=rel_path,
                rule_id=rule_id,
                match_start=ms,
                match_end=me,
                match_kind=m.kind(),
                match_text=m.text(),
                captures=caps,
            ))

    # Deterministic order
    out.sort(key=lambda r: (r.path, r.match_start, r.match_end, r.rule_id))
    return out


def select_rules(rule_pack: dict, rule_ids: List[str]) -> List[dict]:
    all_rules = rule_pack.get("rules") or []
    want = set(rule_ids)
    selected = [r for r in all_rules if r.get("rule_id") in want]
    missing = sorted(want - set(r.get("rule_id") for r in selected))
    if missing:
        raise ValueError(f"Rule ids not found in rule pack: {missing}")
    # Ensure canonical shape
    out = []
    for r in selected:
        if "config" not in r:
            raise ValueError(f"Rule {r.get('rule_id')} missing config")
        out.append({"rule_id": r["rule_id"], "config": r["config"]})
    return out
