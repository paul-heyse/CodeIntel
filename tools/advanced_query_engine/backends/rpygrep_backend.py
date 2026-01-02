"""Ripgrep-backed search helpers for the advanced query engine."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from tools.advanced_query_engine.contracts import QueryBudget, Span
from tools.advanced_query_engine.util.repo_cache import RepoCache

try:
    from rpygrep import RipGrepSearch
except ModuleNotFoundError as exc:  # pragma: no cover
    RipGrepSearch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class RpygrepMatch:
    """Single rpygrep match record."""

    path: str
    pattern_id: str
    line_number: int
    line_text: str
    submatch_start: int
    submatch_end: int
    span: Span | None = None


@dataclass(frozen=True)
class RpygrepResult:
    """Rpygrep execution output."""

    matches: list[RpygrepMatch]
    files_to_patterns: dict[str, list[str]]
    partial: bool
    summary: dict[str, object] | None


@dataclass(frozen=True)
class RpygrepQuery:
    """Inputs for rpygrep candidate execution."""

    repo_root: Path
    preset: dict[str, object]
    pattern_group: dict[str, object]
    budget: QueryBudget
    scope_paths: Iterable[str] | None
    cache: RepoCache | None


@dataclass(frozen=True)
class _PatternSettings:
    preset_opts: dict[str, object]
    include_globs: list[str]
    exclude_globs: list[str]

    @classmethod
    def from_query(cls, query: RpygrepQuery) -> _PatternSettings:
        preset_opts = query.preset.get("options") or {}
        include_globs = list(query.pattern_group.get("globs") or ["**/*.py"])
        exclude_globs = list(query.pattern_group.get("exclude_globs") or [])
        return cls(
            preset_opts=preset_opts,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        )


@dataclass(frozen=True)
class _PatternSpec:
    text: str
    pat_id: str
    is_regex: bool


@dataclass
class _RpygrepState:
    matches: list[RpygrepMatch] = field(default_factory=list)
    files_to_patterns: dict[str, list[str]] = field(default_factory=dict)
    partial: bool = False
    summary: dict[str, object] | None = None


def _load_rg_class() -> type:
    if RipGrepSearch is None:
        msg = (
            "rpygrep is not importable. Install rpygrep and ensure `rg` is in PATH. "
            f"Import error: {_IMPORT_ERROR}"
        )
        raise RuntimeError(msg)
    return RipGrepSearch


def _apply_preset(search: object, preset_opts: dict[str, object], budget: QueryBudget) -> object:
    search = _apply_safe_defaults(search)
    search = _apply_case(search, preset_opts)
    search = _apply_context(search, preset_opts)
    search = _apply_limits(search, preset_opts, budget)
    search = _apply_extra(search, preset_opts)
    return _apply_json(search)


def _apply_safe_defaults(search: object) -> object:
    if hasattr(search, "add_safe_defaults"):
        return search.add_safe_defaults()
    return search


def _apply_case(search: object, preset_opts: dict[str, object]) -> object:
    if "case_sensitive" in preset_opts and hasattr(search, "case_sensitive"):
        return search.case_sensitive(bool(preset_opts["case_sensitive"]))
    return search


def _apply_context(search: object, preset_opts: dict[str, object]) -> object:
    if "before_context" in preset_opts and hasattr(search, "before_context"):
        search = search.before_context(int(preset_opts["before_context"]))
    if "after_context" in preset_opts and hasattr(search, "after_context"):
        search = search.after_context(int(preset_opts["after_context"]))
    return search


def _apply_limits(search: object, preset_opts: dict[str, object], budget: QueryBudget) -> object:
    if hasattr(search, "max_depth") and budget.max_depth:
        search = search.max_depth(int(budget.max_depth))
    if "max_count" in preset_opts and hasattr(search, "max_count"):
        search = search.max_count(int(preset_opts["max_count"]))
    if "max_file_size_bytes" in preset_opts and hasattr(search, "max_file_size"):
        search = search.max_file_size(int(preset_opts["max_file_size_bytes"]))
    return search


def _apply_extra(search: object, preset_opts: dict[str, object]) -> object:
    extra = preset_opts.get("extra_args") or []
    if extra and hasattr(search, "add_extra_options"):
        return search.add_extra_options(list(extra))
    return search


def _apply_json(search: object) -> object:
    if hasattr(search, "as_json"):
        return search.as_json()
    return search


def _apply_scope_targets(search: object, repo_root: Path, scope_paths: Iterable[str]) -> object:
    for rel in scope_paths:
        candidate = (repo_root / rel).resolve()
        if not candidate.exists():
            continue
        if candidate.is_dir() and hasattr(search, "add_directory"):
            search = search.add_directory(candidate)
        elif candidate.is_file() and hasattr(search, "add_file"):
            search = search.add_file(candidate)
    return search


def _apply_globs(
    search: object, include_globs: Iterable[str], exclude_globs: Iterable[str]
) -> object:
    for glob in include_globs:
        if hasattr(search, "include_glob"):
            search = search.include_glob(str(glob))
    for glob in exclude_globs:
        if hasattr(search, "exclude_glob"):
            search = search.exclude_glob(str(glob))
    return search


def _apply_pattern_mode(
    search: object, preset_opts: dict[str, object], *, pattern_is_regex: bool
) -> object:
    if not pattern_is_regex and hasattr(search, "patterns_are_not_regex"):
        return search.patterns_are_not_regex()
    if preset_opts.get("auto_hybrid_regex") and hasattr(search, "auto_hybrid_regex"):
        return search.auto_hybrid_regex()
    return search


def _pattern_entries(pattern_group: dict[str, object]) -> list[dict[str, object]]:
    patterns = pattern_group.get("patterns") or []
    ordered = sorted(
        patterns,
        key=lambda value: (-int(value.get("priority", 0)), str(value.get("pattern"))),
    )
    return [entry for entry in ordered if isinstance(entry, dict)]


def _pattern_specs(pattern_group: dict[str, object]) -> list[_PatternSpec]:
    specs: list[_PatternSpec] = []
    for entry in _pattern_entries(pattern_group):
        pat = entry.get("pattern")
        if not isinstance(pat, str) or not pat:
            continue
        pat_id = str(entry.get("id") or entry.get("pattern_id") or pat)
        specs.append(
            _PatternSpec(text=pat, pat_id=pat_id, is_regex=bool(entry.get("is_regex", True)))
        )
    return specs


def _build_search(
    rg_cls: type,
    query: RpygrepQuery,
    settings: _PatternSettings,
    spec: _PatternSpec,
) -> object:
    search = rg_cls(working_directory=query.repo_root)
    if query.scope_paths:
        search = _apply_scope_targets(search, query.repo_root, query.scope_paths)
    search = _apply_globs(search, settings.include_globs, settings.exclude_globs)
    search = _apply_preset(search, settings.preset_opts, query.budget)
    search = _apply_pattern_mode(search, settings.preset_opts, pattern_is_regex=spec.is_regex)
    if hasattr(search, "add_pattern"):
        search = search.add_pattern(spec.text)
    return search


def _add_file_hit(state: _RpygrepState, budget: QueryBudget, path: str, pat_id: str) -> None:
    limit_reached = budget.max_files and len(state.files_to_patterns) >= budget.max_files
    if limit_reached and path not in state.files_to_patterns:
        state.partial = True
        return
    entries = state.files_to_patterns.setdefault(path, [])
    if pat_id not in entries:
        entries.append(pat_id)


def _handle_match_event(
    query: RpygrepQuery,
    state: _RpygrepState,
    pat_id: str,
    data: dict[str, object],
) -> bool:
    path_text = _event_path(data, query.repo_root)
    if not path_text:
        return False
    _add_file_hit(state, query.budget, path_text, pat_id)
    if state.partial or _budget_exhausted(query.budget, state.matches, state.files_to_patterns):
        state.partial = True
        return True

    line_number, line_text, submatches = _extract_match_payload(data)
    line_index = _match_line_index(query.cache, path_text)
    for start, end in submatches:
        span = _span_from_match(line_index, path_text, line_number, start, end)
        state.matches.append(
            RpygrepMatch(
                path=path_text,
                pattern_id=pat_id,
                line_number=line_number,
                line_text=line_text,
                submatch_start=start,
                submatch_end=end,
                span=span,
            )
        )
        if _budget_exhausted(query.budget, state.matches, state.files_to_patterns):
            state.partial = True
            return True
    return False


def _collect_matches_for_pattern(
    query: RpygrepQuery,
    state: _RpygrepState,
    search: object,
    pat_id: str,
) -> None:
    if not hasattr(search, "run_direct"):
        return
    for line in search.run_direct():  # type: ignore[attr-defined]
        payload = _parse_json_line(line)
        if payload is None:
            continue
        event_type = payload.get("type")
        if event_type == "summary":
            data = payload.get("data")
            if isinstance(data, dict):
                state.summary = data
            continue
        if event_type != "match":
            continue
        data = payload.get("data") or {}
        if _handle_match_event(query, state, pat_id, data):
            break


def run_pattern_group(query: RpygrepQuery) -> RpygrepResult:
    """Execute a rpygrep pattern group and return normalized results.

    Returns
    -------
    RpygrepResult
        Collected rpygrep matches.
    """
    rg_cls = _load_rg_class()
    settings = _PatternSettings.from_query(query)
    state = _RpygrepState()

    for spec in _pattern_specs(query.pattern_group):
        search = _build_search(rg_cls, query, settings, spec)
        _collect_matches_for_pattern(query, state, search, spec.pat_id)
        if state.partial:
            break

    state.matches.sort(
        key=lambda match: (
            match.path,
            match.line_number,
            match.submatch_start,
            match.submatch_end,
            match.pattern_id,
        )
    )

    return RpygrepResult(
        matches=state.matches,
        files_to_patterns=state.files_to_patterns,
        partial=state.partial,
        summary=state.summary,
    )


def _parse_json_line(line: str) -> dict[str, object] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _event_path(data: dict[str, object], repo_root: Path) -> str | None:
    path_obj = data.get("path")
    if isinstance(path_obj, dict) and isinstance(path_obj.get("text"), str):
        text = path_obj.get("text")
        if not text:
            return None
        path = Path(text)
        if path.is_absolute():
            try:
                return path.relative_to(repo_root).as_posix()
            except ValueError:
                return None
        return path.as_posix()
    return None


def _extract_match_payload(data: dict[str, object]) -> tuple[int, str, list[tuple[int, int]]]:
    line_number = int(data.get("line_number") or 0)
    lines_obj = data.get("lines") or {}
    line_text = ""
    if isinstance(lines_obj, dict) and isinstance(lines_obj.get("text"), str):
        line_text = lines_obj.get("text", "").rstrip("\n")
    submatches = _submatches(data.get("submatches"), len(line_text))
    return line_number, line_text, submatches


def _submatches(value: object, line_len: int) -> list[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        return [(0, max(0, line_len))]
    spans: list[tuple[int, int]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        start = int(entry.get("start") or 0)
        end = int(entry.get("end") or start)
        spans.append((start, end))
    return spans


def _match_line_index(cache: RepoCache | None, path_text: str) -> object | None:
    if cache is None:
        return None
    try:
        return cache.line_index(path_text)
    except FileNotFoundError:
        return None


def _span_from_match(
    line_index: object | None,
    path: str,
    line_number: int,
    start: int,
    end: int,
) -> Span | None:
    if line_index is None or line_number <= 0:
        return None
    line_start = line_index.line_start_byte(line_number)
    return Span(
        path=path,
        start_byte=line_start + start,
        end_byte=line_start + end,
        **line_index.span_to_range(line_start + start, line_start + end),
    )


def _budget_exhausted(
    budget: QueryBudget,
    matches: list[RpygrepMatch],
    files_to_patterns: dict[str, list[str]],
) -> bool:
    return (budget.max_matches and len(matches) >= budget.max_matches) or (
        budget.max_files and len(files_to_patterns) >= budget.max_files
    )


__all__ = ["RpygrepMatch", "RpygrepQuery", "RpygrepResult", "run_pattern_group"]
