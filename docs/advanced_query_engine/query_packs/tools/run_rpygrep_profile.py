\
#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rpygrep import DEFAULT_EXCLUDED_TYPES, RipGrepSearch


def _load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _maybe_text(lines_obj: Any) -> str | None:
    # rpygrep matches ripgrep JSON: content can be text or bytes; text may be None.
    txt = getattr(lines_obj, "text", None)
    if isinstance(txt, str):
        return txt.rstrip("\n")
    return None


def run_pattern_group(*, repo_root: Path, preset_file: Path, pattern_group_file: Path) -> dict[str, Any]:
    preset = _load_json(preset_file)
    pg = _load_json(pattern_group_file)

    opts = preset["options"]
    rg = (
        RipGrepSearch(working_directory=repo_root)
        .add_safe_defaults()
        .exclude_types(tuple(DEFAULT_EXCLUDED_TYPES))
    )

    # Globs
    for g in opts.get("exclude_globs", []):
        rg.exclude_glob(g)
    for g in pg.get("exclude_globs", []):
        rg.exclude_glob(g)
    for g in pg.get("globs", []):
        rg.include_glob(g)

    # Output shaping
    rg.case_sensitive(bool(opts.get("case_sensitive", True)))
    if "before_context" in opts:
        rg.before_context(int(opts["before_context"]))
    if "after_context" in opts:
        rg.after_context(int(opts["after_context"]))
    if "max_count" in opts:
        rg.max_count(int(opts["max_count"]))
    if "max_file_size_bytes" in opts:
        rg.max_file_size(int(opts["max_file_size_bytes"]))

    # Regex engine behavior
    if opts.get("patterns_are_not_regex", False):
        rg.patterns_are_not_regex()
    if opts.get("auto_hybrid_regex", False):
        rg.auto_hybrid_regex()

    # Extra raw rg flags
    extra = opts.get("extra_args", [])
    if extra:
        rg.add_extra_options(list(extra))

    # Patterns
    for p in sorted(pg["patterns"], key=lambda x: (-int(x.get("priority", 0)), x["pattern"])):
        pat = p["pattern"]
        if not p.get("is_regex", True):
            # If we are NOT in fixed-string mode, escape literal patterns.
            if not opts.get("patterns_are_not_regex", False):
                pat = re.escape(pat)
        rg.add_pattern(pat)

    # Execute
    out_files: list[dict[str, Any]] = []
    for r in rg.run():
        path = getattr(r, "path", None)
        path_s = str(path) if path is not None else "<unknown>"

        matches_out: list[dict[str, Any]] = []
        for m in getattr(r, "matches", []) or []:
            data = getattr(m, "data", None)
            ln = getattr(data, "line_number", None)
            lines = getattr(data, "lines", None)
            line_txt = _maybe_text(lines) if lines is not None else None

            sub_out: list[dict[str, Any]] = []
            for sm in getattr(data, "submatches", []) or []:
                sub_out.append({"start": getattr(sm, "start", None), "end": getattr(sm, "end", None)})

            matches_out.append({"line_number": ln, "line": line_txt if line_txt is not None else "<BINARY>", "submatches": sub_out})

        ctx_out: list[dict[str, Any]] = []
        for c in getattr(r, "context", []) or []:
            data = getattr(c, "data", None)
            ln = getattr(data, "line_number", None)
            lines = getattr(data, "lines", None)
            line_txt = _maybe_text(lines) if lines is not None else None
            ctx_out.append({"line_number": ln, "line": line_txt if line_txt is not None else "<BINARY>"})

        # Stable ordering inside a file
        matches_out.sort(key=lambda x: (x["line_number"] or 0, (x["submatches"][0]["start"] if x["submatches"] else -1)))
        ctx_out.sort(key=lambda x: (x["line_number"] or 0))

        out_files.append({"path": path_s, "matches": matches_out, "context": ctx_out})

    # Stable ordering across files
    out_files.sort(key=lambda x: x["path"])

    return {"preset_id": preset["preset_id"], "pattern_group_id": pg["pattern_group_id"], "files": out_files}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, required=True)
    ap.add_argument("--preset", type=str, required=True)
    ap.add_argument("--patterns", type=str, required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()
    preset_file = Path(args.preset).resolve()
    pattern_group_file = Path(args.patterns).resolve()

    result = run_pattern_group(repo_root=repo_root, preset_file=preset_file, pattern_group_file=pattern_group_file)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
