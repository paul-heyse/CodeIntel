from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import json

# rpygrep is a thin wrapper around ripgrep; it requires `rg` in PATH.
# We intentionally use run_direct() to parse raw JSON lines ourselves (stable contract).
try:
    from rpygrep import RipGrepSearch, DEFAULT_EXCLUDED_TYPES  # type: ignore
except Exception as e:  # pragma: no cover
    RipGrepSearch = None
    DEFAULT_EXCLUDED_TYPES = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class RgHit:
    path: str
    pattern_id: str
    line_number: int
    line_text: str
    submatches: list[dict]  # {start,end,match_text}
    # raw JSON event provenance
    rg_type: str


@dataclass
class RgCandidates:
    # file -> list of pattern_ids that hit it
    files_to_patterns: Dict[str, List[str]]
    hits: List[RgHit]
    partial: bool
    summary: Optional[dict]


def _apply_preset(search: Any, preset_opts: dict, pattern_is_regex: bool) -> Any:
    """Apply preset options onto a RipGrepSearch builder."""
    # Safe defaults are important: depth caps, size caps, etc.
    if hasattr(search, "add_safe_defaults"):
        search = search.add_safe_defaults()

    # Exclude common binary/vendor types
    if DEFAULT_EXCLUDED_TYPES is not None and hasattr(search, "exclude_types"):
        search = search.exclude_types(DEFAULT_EXCLUDED_TYPES)

    # Globs
    for g in preset_opts.get("exclude_globs", []) or []:
        if hasattr(search, "exclude_glob"):
            search = search.exclude_glob(g)

    # Core knobs
    if "case_sensitive" in preset_opts and hasattr(search, "case_sensitive"):
        search = search.case_sensitive(bool(preset_opts["case_sensitive"]))

    # Regex vs literal:
    # By default, ripgrep treats patterns as regex. For literal patterns, flip the mode.
    if (not pattern_is_regex) and preset_opts.get("patterns_are_not_regex") is not False:
        # If preset says patterns_are_not_regex True (or absent), set literal mode.
        if hasattr(search, "patterns_are_not_regex"):
            search = search.patterns_are_not_regex()
    elif (not pattern_is_regex) and hasattr(search, "patterns_are_not_regex"):
        # pattern says literal but preset explicitly requests regex -> still switch to literal
        search = search.patterns_are_not_regex()

    # Context lines
    if "before_context" in preset_opts and hasattr(search, "before_context"):
        search = search.before_context(int(preset_opts["before_context"]))
    if "after_context" in preset_opts and hasattr(search, "after_context"):
        search = search.after_context(int(preset_opts["after_context"]))

    # Output bounds
    if "max_count" in preset_opts and hasattr(search, "max_count"):
        search = search.max_count(int(preset_opts["max_count"]))
    if "max_file_size_bytes" in preset_opts and hasattr(search, "max_file_size"):
        search = search.max_file_size(int(preset_opts["max_file_size_bytes"]))

    # Extra args
    extra = preset_opts.get("extra_args", []) or []
    if extra and hasattr(search, "add_extra_options"):
        search = search.add_extra_options(list(extra))

    return search


def run_candidates(
    *,
    repo_root: Path,
    preset: dict,
    pattern_group: dict,
    # guardrails
    hard_max_files: Optional[int] = None,
) -> RgCandidates:
    """Run rpygrep candidate stage (required by wiring packs).

    We run each pattern independently to preserve per-pattern regex/literal semantics.
    We parse raw ripgrep JSON (run_direct) so we do not depend on rpygrep's internal model.
    """
    if RipGrepSearch is None:
        raise RuntimeError(
            "rpygrep is not importable. Install rpygrep and ensure `rg` is in PATH. " 
            f"Import error: {_IMPORT_ERROR}"
        )

    preset_opts = (preset.get("options") or {})
    patterns = pattern_group.get("patterns") or []
    include_globs = pattern_group.get("globs") or ["**/*.py"]
    exclude_globs = pattern_group.get("exclude_globs") or []

    files_to_patterns: Dict[str, List[str]] = {}
    hits: List[RgHit] = []
    partial = False
    summary: Optional[dict] = None

    # Track file budget and stop early if exceeded.
    def add_file_hit(path: str, pat_id: str) -> None:
        nonlocal partial
        if hard_max_files is not None and len(files_to_patterns) >= hard_max_files and path not in files_to_patterns:
            partial = True
            return
        files_to_patterns.setdefault(path, [])
        if pat_id not in files_to_patterns[path]:
            files_to_patterns[path].append(pat_id)

    for p in patterns:
        pat = p.get("pattern")
        if not pat:
            continue
        pat_id = p.get("id") or p.get("pattern_id") or p.get("pattern")  # stable enough
        is_regex = bool(p.get("is_regex", True))

        search = RipGrepSearch(working_directory=repo_root)

        # include/exclude globs from group
        for g in include_globs:
            if hasattr(search, "include_glob"):
                search = search.include_glob(g)
        for g in exclude_globs:
            if hasattr(search, "exclude_glob"):
                search = search.exclude_glob(g)

        search = _apply_preset(search, preset_opts, pattern_is_regex=is_regex)

        # Pattern registration
        if hasattr(search, "add_pattern"):
            search = search.add_pattern(pat)

        # Execute raw-json mode
        # NOTE: run_direct() returns json lines (str)
        for line in search.run_direct():  # type: ignore[attr-defined]
            try:
                ev = json.loads(line)
            except Exception:
                continue
            t = ev.get("type")
            data = ev.get("data") or {}
            if t == "match":
                path = (data.get("path") or {}).get("text") or ""
                if not path:
                    continue
                add_file_hit(path, pat_id)
                if partial and (hard_max_files is not None and len(files_to_patterns) >= hard_max_files):
                    break
                line_number = int(data.get("line_number") or 0)
                line_text = ((data.get("lines") or {}).get("text") or "").rstrip("\n")
                subs = []
                for sm in (data.get("submatches") or []):
                    subs.append({
                        "start": int(sm.get("start") or 0),
                        "end": int(sm.get("end") or 0),
                        "match_text": ((sm.get("match") or {}).get("text") or ""),
                    })
                hits.append(RgHit(
                    path=path,
                    pattern_id=pat_id,
                    line_number=line_number,
                    line_text=line_text,
                    submatches=subs,
                    rg_type="match",
                ))
            elif t == "summary":
                summary = data
            # Ignore context/begin/end by default; evidence is later derived from file bytes.

    return RgCandidates(files_to_patterns=files_to_patterns, hits=hits, partial=partial, summary=summary)
