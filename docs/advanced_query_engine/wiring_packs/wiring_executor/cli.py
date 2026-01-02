from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

from .executor import execute_pack, execute_packs


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="wiring-exec", description="Execute Python wiring packs and emit wiring edges JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run one wiring pack")
    p_run.add_argument("--repo", required=True, help="Path to repo root")
    p_run.add_argument("--pack", required=True, help="Path to wiring pack JSON spec")
    p_run.add_argument("--pack-root", default=None, help="Root dir for resolving pack-relative files (defaults to pack dir)")
    p_run.add_argument("--out", default=None, help="Output JSON path (defaults to stdout)")
    p_run.add_argument("--no-cross-file-resolve", action="store_true", help="Disable cross-file handler resolution")
    p_run.add_argument("--max-candidate-files", type=int, default=800, help="Hard cap for candidate files from rpygrep")

    p_all = sub.add_parser("run-all", help="Run multiple wiring packs")
    p_all.add_argument("--repo", required=True)
    p_all.add_argument("--packs", nargs="+", required=True, help="List of wiring pack JSON spec files")
    p_all.add_argument("--pack-root", default=None)
    p_all.add_argument("--out", default=None)
    p_all.add_argument("--no-cross-file-resolve", action="store_true")
    p_all.add_argument("--max-candidate-files", type=int, default=800)

    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    ns = _parse_args(argv)

    if ns.cmd == "run":
        res = execute_pack(
            repo_root=ns.repo,
            pack_file=ns.pack,
            pack_root=ns.pack_root,
            allow_cross_file_handler_resolution=not ns.no_cross_file_resolve,
            hard_max_candidate_files=ns.max_candidate_files,
        )
    else:
        res = execute_packs(
            repo_root=ns.repo,
            packs=ns.packs,
            pack_root=ns.pack_root,
            allow_cross_file_handler_resolution=not ns.no_cross_file_resolve,
            hard_max_candidate_files=ns.max_candidate_files,
        )

    out = json.dumps(res, indent=2, ensure_ascii=False)
    if ns.out:
        Path(ns.out).write_text(out, encoding="utf-8")
    else:
        print(out)


if __name__ == "__main__":
    main()
