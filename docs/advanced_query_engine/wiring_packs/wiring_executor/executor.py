from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import ast
import json
import re

from .packs import PackPaths, load_json, load_preset, load_pattern_group, load_ast_grep_rules
from .engines.rpygrep_runner import run_candidates, RgCandidates
from .engines.astgrep_runner import select_rules, run_ast_grep_rules_on_file, AGMatchRecord, CaptureValue
from .enrich.libcst_defs import build_def_index, DefIndex, DefRec
from .util.hashing import stable_hex_digest
from .util.template import safe_format
from .util.line_index import LineIndex
from .util.snippets import build_evidence


@dataclass
class RepoCache:
    repo_root: Path
    _bytes: Dict[str, bytes]
    _line_index: Dict[str, LineIndex]
    _def_index: Dict[str, DefIndex]

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._bytes = {}
        self._line_index = {}
        self._def_index = {}

    def read_bytes(self, rel_path: str) -> bytes:
        if rel_path not in self._bytes:
            self._bytes[rel_path] = (self.repo_root / rel_path).read_bytes()
        return self._bytes[rel_path]

    def line_index(self, rel_path: str) -> LineIndex:
        if rel_path not in self._line_index:
            self._line_index[rel_path] = LineIndex.build(self.read_bytes(rel_path))
        return self._line_index[rel_path]

    def def_index(self, rel_path: str) -> DefIndex:
        if rel_path not in self._def_index:
            self._def_index[rel_path] = build_def_index(rel_path, self.read_bytes(rel_path))
        return self._def_index[rel_path]


def _capture_text(rec: AGMatchRecord, name: str) -> Optional[str]:
    vals = rec.captures.get(name)
    if not vals:
        return None
    # for single captures, prefer first; for multi captures, join w/ comma
    if len(vals) == 1:
        return vals[0].text
    return ", ".join(v.text for v in vals)


def _unquote_python_string_literal(s: str) -> Optional[str]:
    s = s.strip()
    if len(s) < 2:
        return None
    if s[0] not in ('"', "'"):
        return None
    try:
        v = ast.literal_eval(s)
        if isinstance(v, str):
            return v
    except Exception:
        return None
    return None


def _normalize_method(m: Optional[str]) -> str:
    if not m:
        return "*"
    m = m.strip()
    # If capture comes from attribute like get/post, it is already lowercase.
    return m.upper()


def _simple_name(expr: str) -> Optional[str]:
    expr = expr.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
        return expr
    return None


def _span_dict_from_capture(path: str, cap: CaptureValue, li: LineIndex) -> dict:
    # Convert ast-grep 0-indexed line/col to canonical range and byte span.
    # Prefer byte spans as truth.
    r = li.span_to_range(cap.start_byte, cap.end_byte)
    return {"path": path, "start_byte": cap.start_byte, "end_byte": cap.end_byte, **r}


def _derive_handler_from_enclosing_def(cache: RepoCache, rec: AGMatchRecord) -> Optional[DefRec]:
    try:
        idx = cache.def_index(rec.path)
    except Exception:
        return None
    return idx.enclosing_def(rec.match_start)


def _resolve_handler_capture(
    cache: RepoCache,
    repo_root: Path,
    handler_expr: str,
    *,
    local_file: str,
    allow_cross_file_resolution: bool,
    rpygrep_preset_for_resolution: dict,
) -> Tuple[Optional[DefRec], Dict[str, Any]]:
    """Attempt to resolve a handler expression to a DefRec.

    Strategy:
      1) If handler is a simple name and exists in same file -> use it.
      2) If allow_cross_file_resolution, use rpygrep to find candidate 'def <name>' sites,
         then confirm with LibCST def index.

    Returns (defrec or None, debug dict).
    """
    debug: Dict[str, Any] = {"handler_expr": handler_expr, "strategy": None, "candidates": []}

    name = _simple_name(handler_expr)
    if not name:
        debug["strategy"] = "non_simple_expr"
        return None, debug

    debug["strategy"] = "local_then_global"

    try:
        local_idx = cache.def_index(local_file)
        cands = local_idx.by_name(name)
        if cands:
            debug["strategy"] = "local"
            return cands[0], debug
    except Exception:
        pass

    if not allow_cross_file_resolution:
        debug["strategy"] = "unresolved_no_global"
        return None, debug

    # Global: rpygrep for function defs. This is bounded and best-effort.
    # Regex is used here; do not run patterns_are_not_regex.
    from .engines.rpygrep_runner import run_candidates  # local import to keep module importable without rpygrep
    # Build a minimal in-memory pattern group just for resolution.
    pat = rf"\\bdef\\s+{re.escape(name)}\\b"
    pattern_group = {
        "pattern_group_id": "rg.python.resolve.handler",
        "patterns": [{"id": f"def:{name}", "pattern": pat, "is_regex": True}],
        "globs": ["**/*.py"],
        "exclude_globs": ["**/.venv/**", "**/venv/**", "**/site-packages/**"],
    }
    preset = rpygrep_preset_for_resolution
    rg = run_candidates(repo_root=repo_root, preset=preset, pattern_group=pattern_group, hard_max_files=30)
    debug["rg_files"] = list(rg.files_to_patterns.keys())
    # Confirm by parsing only those files.
    for rel in rg.files_to_patterns.keys():
        try:
            idx = cache.def_index(rel)
            cands = idx.by_name(name)
            for c in cands:
                debug["candidates"].append({"path": rel, "qname": c.qname, "start": c.start_byte, "end": c.end_byte})
        except Exception:
            continue
    if debug["candidates"]:
        best = debug["candidates"][0]
        try:
            idx = cache.def_index(best["path"])
            return idx.by_name(name)[0], debug
        except Exception:
            return None, debug
    return None, debug


def _postprocess_records(pack: dict, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply pack-defined postprocess operations.

    Supported ops:
      - python.unquote_capture: capture_names[], output_field_suffix (default: _unquoted)
      - python.normalize_http_method: capture_name, output_field (default: http_method)
      - python.upper_capture: capture_name, output_field
      - python.join_argparse_subcommands: joins add_parser + set_defaults to produce command names
    """
    ops = pack.get("postprocess") or []

    # Split into record-level and batch-level ops.
    batch_ops = [op for op in ops if op.get("op") == "python.join_argparse_subcommands"]
    rec_ops = [op for op in ops if op.get("op") != "python.join_argparse_subcommands"]

    # Record-level ops
    for op in rec_ops:
        name = op.get("op")
        if name == "python.unquote_capture":
            suffix = op.get("output_field_suffix") or "_unquoted"
            for r in records:
                for cap in (op.get("capture_names") or []):
                    val = r.get("captures", {}).get(cap)
                    if val is None:
                        continue
                    uq = _unquote_python_string_literal(str(val))
                    if uq is not None:
                        r[f"{cap}{suffix}"] = uq
        elif name == "python.normalize_http_method":
            cap = op.get("capture_name") or "METHOD"
            out = op.get("output_field") or "http_method"
            for r in records:
                r[out] = _normalize_method(r.get("captures", {}).get(cap))
        elif name == "python.upper_capture":
            cap = op.get("capture_name")
            out = op.get("output_field") or cap
            if not cap:
                continue
            for r in records:
                v = r.get("captures", {}).get(cap)
                if v is not None:
                    r[out] = str(v).upper()
        else:
            # unknown op -> ignore for forward compatibility
            continue

    # Batch-level ops: argparse join
    for op in batch_ops:
        # expected capture names
        sub_var = op.get("subparser_var_capture", "SUB")
        cmd_cap = op.get("command_capture", "CMD")
        handler_cap = op.get("handler_capture", "HANDLER")
        # Build file-local maps: SUB -> CMD
        by_file: Dict[str, Dict[str, str]] = {}
        for r in records:
            if r.get("rule_id") == op.get("add_parser_rule_id", "py.argparse.add_parser.assign"):
                sub = (r.get("captures", {}) or {}).get(sub_var)
                cmd = (r.get("captures", {}) or {}).get(cmd_cap)
                if sub and cmd:
                    by_file.setdefault(r["path"], {})[str(sub)] = r.get(f"{cmd_cap}_unquoted") or str(cmd)
        # Annotate set_defaults records
        for r in records:
            if r.get("rule_id") == op.get("set_defaults_rule_id", "py.argparse.subparser.set_defaults"):
                sub = (r.get("captures", {}) or {}).get(sub_var)
                if sub:
                    cmd = by_file.get(r["path"], {}).get(str(sub))
                    if cmd:
                        r["ARGPARSE_CMD"] = cmd

    return records


def _emit_edges(
    *,
    pack: dict,
    cache: RepoCache,
    repo_root: Path,
    rg_candidates: RgCandidates,
    ag_matches: List[AGMatchRecord],
    allow_cross_file_handler_resolution: bool,
    handler_resolution_preset: dict,
) -> Dict[str, Any]:
    """Transform raw matches into wiring edges."""
    # Build intermediate record dicts
    recs: List[Dict[str, Any]] = []
    for m in ag_matches:
        li = cache.line_index(m.path)
        cap_texts: Dict[str, Any] = {}
        cap_spans: Dict[str, Any] = {}
        for k, vals in (m.captures or {}).items():
            if not vals:
                continue
            # store primary text
            if len(vals) == 1:
                cap_texts[k] = vals[0].text
                cap_spans[k] = _span_dict_from_capture(m.path, vals[0], li)
            else:
                cap_texts[k] = [v.text for v in vals]
                cap_spans[k] = [_span_dict_from_capture(m.path, v, li) for v in vals]

        recs.append({
            "path": m.path,
            "rule_id": m.rule_id,
            "match_span": {"path": m.path, "start_byte": m.match_start, "end_byte": m.match_end, **li.span_to_range(m.match_start, m.match_end)},
            "captures": cap_texts,
            "capture_spans": cap_spans,
            "rg_pattern_ids": rg_candidates.files_to_patterns.get(m.path, []),
        })

    # Enrichment: derive enclosing handler for each match
    for r in recs:
        try:
            enc = cache.def_index(r["path"]).enclosing_def(int(r["match_span"]["start_byte"]))
        except Exception:
            enc = None
        if enc:
            li_enc = cache.line_index(enc.path)
            r["enclosing_def"] = {
                "kind": enc.kind,
                "name": enc.name,
                "qname": enc.qname,
                "def_span": {
                    "path": enc.path,
                    "start_byte": enc.start_byte,
                    "end_byte": enc.end_byte,
                    **li_enc.span_to_range(enc.start_byte, enc.end_byte),
                },
            }

    # Apply postprocess
    recs = _postprocess_records(pack, recs)

    # Emit edges
    emit = pack.get("emit") or {}
    template_default = emit.get("entry_key_template") or "{pack_id}:{path}:{rule_id}"
    entry_key_by_rule = emit.get("entry_key_by_rule") or {}
    target_hint_by_rule = emit.get("target_symbol_hint_by_rule") or {}
    hook_span_by_rule = emit.get("hook_span_by_rule") or {}
    entry_kind = pack.get("entry_kind") or pack.get("entryKind") or "wiring"
    pack_id = pack.get("pack_id") or "<unknown>"

    edges: List[Dict[str, Any]] = []
    for r in recs:
        # Determine hook span: default to MATCH_NODE if present
        hook_cap = hook_span_by_rule.get(r.get("rule_id")) or (emit.get("hook_span_capture") or "MATCH_NODE")
        hook_span = (r.get("capture_spans") or {}).get(hook_cap) or r.get("match_span")

        # Determine target: prefer explicit handler capture, else enclosing def
        handler_expr = None
        target_hint = target_hint_by_rule.get(r.get("rule_id")) or emit.get("target_symbol_hint_capture")
        for cap in [target_hint, "HANDLER", "handler", "DEP"]:
            if cap and (r.get("captures") or {}).get(cap) is not None:
                handler_expr = (r.get("captures") or {}).get(cap)
                break

        target = r.get("enclosing_def")
        resolution_debug = None
        if handler_expr is not None and isinstance(handler_expr, str):
            resolved, dbg = _resolve_handler_capture(
                cache,
                repo_root,
                handler_expr,
                local_file=r["path"],
                allow_cross_file_resolution=allow_cross_file_handler_resolution,
                rpygrep_preset_for_resolution=handler_resolution_preset,
            )
            resolution_debug = dbg
            if resolved is not None:
                target = {
                    "kind": resolved.kind,
                    "name": resolved.name,
                    "qname": resolved.qname,
                    "def_span": {"path": resolved.path, "start_byte": resolved.start_byte, "end_byte": resolved.end_byte, **cache.line_index(resolved.path).span_to_range(resolved.start_byte, resolved.end_byte)},
                }
        # Build entry key values
        fmt_values: Dict[str, Any] = {}
        fmt_values.update(r.get("captures") or {})
        # derived convenience
        if "PATH_unquoted" not in fmt_values and (r.get("PATH_unquoted") is not None):
            fmt_values["PATH_unquoted"] = r.get("PATH_unquoted")
        fmt_values.update({k: v for k, v in r.items() if k.endswith("_unquoted")})
        # enclosing def convenience
        enc = r.get("enclosing_def") or {}
        if isinstance(enc, dict):
            fmt_values.setdefault("enclosing_name", enc.get("name"))
            fmt_values.setdefault("enclosing_qname", enc.get("qname"))

        fmt_values.setdefault("http_method", r.get("http_method"))
        fmt_values.setdefault("pack_id", pack_id)
        fmt_values.setdefault("path", r.get("path"))
        fmt_values.setdefault("rule_id", r.get("rule_id"))
        if r.get("ARGPARSE_CMD") is not None:
            fmt_values["ARGPARSE_CMD"] = r["ARGPARSE_CMD"]

        tmpl = entry_key_by_rule.get(r.get("rule_id")) or template_default
        entry_key = safe_format(tmpl, fmt_values)

        # Evidence from source bytes
        src = cache.read_bytes(r["path"])
        ev = build_evidence(src, r["path"], int(hook_span["start_byte"]), int(hook_span["end_byte"]), before_lines=1, after_lines=1)

        edge_id = stable_hex_digest(pack_id, entry_kind, entry_key, hook_span.get("path"), hook_span.get("start_byte"), hook_span.get("end_byte"), (target or {}).get("qname"))

        edges.append({
            "edge_id": edge_id,
            "pack_id": pack_id,
            "framework": pack.get("framework"),
            "entry_kind": entry_kind,
            "entry_key": entry_key,
            "hook_span": hook_span,
            "target": target,
            "match": {
                "path": r.get("path"),
                "rule_id": r.get("rule_id"),
                "match_span": r.get("match_span"),
                "captures": r.get("captures"),
                "capture_spans": r.get("capture_spans"),
                "rg_pattern_ids": r.get("rg_pattern_ids"),
                "handler_resolution": resolution_debug,
            },
            "evidence": {
                "span": ev.span,
                "excerpt": ev.excerpt,
                "context": ev.context,
            },
        })

    edges.sort(key=lambda e: (e["entry_kind"], e["entry_key"], e["hook_span"]["path"], e["hook_span"]["start_byte"]))
    return {
        "pack_id": pack_id,
        "entry_kind": entry_kind,
        "framework": pack.get("framework"),
        "partial": bool(rg_candidates.partial),
        "edges": edges,
        "debug": {
            "rg_files": sorted(rg_candidates.files_to_patterns.keys()),
            "rg_hit_count": len(rg_candidates.hits),
            "rg_partial": rg_candidates.partial,
        }
    }


def execute_pack(
    *,
    repo_root: str | Path,
    pack_file: str | Path,
    pack_root: Optional[str | Path] = None,
    allow_cross_file_handler_resolution: bool = True,
    handler_resolution_preset_id: str = "rg.audit_deterministic",
    hard_max_candidate_files: int = 800,
) -> Dict[str, Any]:
    """Execute a single wiring pack and return a wiring map.

    Parameters
    ----------
    repo_root:
      Filesystem path to the target repo root.

    pack_file:
      Path to wiring pack JSON spec.

    pack_root:
      Directory used to resolve relative references inside the pack spec.
      Defaults to the parent dir of pack_file.

    allow_cross_file_handler_resolution:
      If True, handler expressions like HANDLER=foo will be resolved across repo using a bounded rpygrep search.

    handler_resolution_preset_id:
      rpygrep preset id used for cross-file resolution; should be more deterministic than interactive presets.

    hard_max_candidate_files:
      Hard cap on unique files returned by rpygrep candidate stage.
    """
    repo_root = Path(repo_root).resolve()
    pack_file = Path(pack_file).resolve()
    pack_root_path = Path(pack_root).resolve() if pack_root else pack_file.parent.resolve()
    if pack_root is None:
        # Auto-detect pack root by walking upward looking for the rpygrep presets folder.
        cur = pack_root_path
        found = None
        for _ in range(6):
            if (cur / "rpygrep" / "presets").exists():
                found = cur
                break
            if cur.parent == cur:
                break
            cur = cur.parent
        if found is not None:
            pack_root_path = found

    pack_paths = PackPaths(pack_root=pack_root_path)

    pack = load_json(pack_file)

    stages = pack.get("stages") or []
    if not stages or stages[0].get("engine") != "rpygrep":
        raise ValueError("All wiring packs must start with an rpygrep candidate stage")

    cache = RepoCache(repo_root)

    # --- Stage 1: rpygrep candidates
    rg_stage = stages[0]
    preset = load_preset(pack_paths, rg_stage["preset"])
    pattern_group = load_pattern_group(pack_paths, rg_stage["pattern_group_file"])
    rg = run_candidates(repo_root=repo_root, preset=preset, pattern_group=pattern_group, hard_max_files=hard_max_candidate_files)

    candidate_files = sorted(rg.files_to_patterns.keys())

    # --- Stage 2: ast-grep (required for these packs)
    ag_matches: List[AGMatchRecord] = []
    for st in stages[1:]:
        if st.get("engine") != "ast_grep":
            continue
        rule_pack = load_ast_grep_rules(pack_paths, st["rules_file"])
        rules = select_rules(rule_pack, st.get("rule_ids") or [])
        for rel in candidate_files:
            # Ignore non-python files defensively
            if not rel.endswith(".py"):
                continue
            try:
                src = (repo_root / rel).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                src = (repo_root / rel).read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                continue
            try:
                ag_matches.extend(run_ast_grep_rules_on_file(rel_path=rel, src_text=src, rules=rules))
            except Exception:
                # Structural stage should be best-effort; keep going.
                continue

    # Handler-resolution uses a more deterministic preset
    handler_preset = load_preset(pack_paths, handler_resolution_preset_id)

    return _emit_edges(
        pack=pack,
        cache=cache,
        repo_root=repo_root,
        rg_candidates=rg,
        ag_matches=ag_matches,
        allow_cross_file_handler_resolution=allow_cross_file_handler_resolution,
        handler_resolution_preset=handler_preset,
    )


def execute_packs(
    *,
    repo_root: str | Path,
    packs: List[str | Path],
    pack_root: Optional[str | Path] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute multiple pack files and return aggregated edges."""
    repo_root = Path(repo_root).resolve()
    results: List[Dict[str, Any]] = []
    for p in packs:
        res = execute_pack(repo_root=repo_root, pack_file=p, pack_root=pack_root, **kwargs)
        results.append(res)
    # Flatten edges and keep per-pack sections.
    all_edges: List[Dict[str, Any]] = []
    for r in results:
        all_edges.extend(r.get("edges") or [])
    all_edges.sort(key=lambda e: (e["entry_kind"], e["entry_key"]))
    return {
        "repo_root": str(repo_root),
        "packs": [r.get("pack_id") for r in results],
        "edges": all_edges,
        "by_pack": results,
    }
