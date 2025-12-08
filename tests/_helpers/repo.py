"""Canonical repository fixtures for tests.

Writes a standard pkg/ package and exposes module paths and GOID mappings
shared across seed packs and graph/coverage helpers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Module paths and FQNs
MOD_A_PATH: Final[str] = "pkg/mod_a.py"
MOD_B_PATH: Final[str] = "pkg/mod_b.py"
MOD_C_PATH: Final[str] = "pkg/mod_c.py"
MOD_UTIL_PATH: Final[str] = "pkg/util.py"

MOD_A_FQN: Final[str] = "pkg.mod_a"
MOD_B_FQN: Final[str] = "pkg.mod_b"
MOD_C_FQN: Final[str] = "pkg.mod_c"
MOD_UTIL_FQN: Final[str] = "pkg.util"

# GOID assignments (kept stable for cross-pack references)
GOID_FUNC_A: Final[int] = 1001
GOID_FUNC_B: Final[int] = 1002
GOID_FUNC_C: Final[int] = 1003
GOID_HELPER: Final[int] = 1004
GOID_CALLER: Final[int] = 1005
GOID_CALLEE: Final[int] = 1006


@dataclass(frozen=True)
class CanonicalFunction:
    """Function metadata for canonical repo fixtures."""

    qualname: str
    goid: int
    rel_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CanonicalRepo:
    """Canonical repo layout including module paths and GOID metadata."""

    module_paths: dict[str, str]
    goids: dict[str, int]
    functions: dict[str, CanonicalFunction]


def write_canonical_repo(repo_root: Path) -> CanonicalRepo:
    """Write canonical pkg/ modules and return metadata."""
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    module_sources = {
        MOD_A_PATH: "\n".join(
            [
                "def func_a(x: int, y: int) -> int:",
                "    from pkg.mod_b import func_b",
                "    return func_b(x) + y",
            ]
        ),
        MOD_B_PATH: "\n".join(
            [
                "def func_b(x: int) -> int:",
                "    from pkg.mod_c import func_c",
                "    total = x * 2",
                "    for value in func_c():",
                "        total += value",
                "    return total",
            ]
        ),
        MOD_C_PATH: "\n".join(
            [
                "def func_c() -> range:",
                "    return range(1, 4)",
            ]
        ),
        MOD_UTIL_PATH: "\n".join(
            [
                "def helper(value: int) -> int:",
                "    return value",
            ]
        ),
    }
    written_paths: dict[str, Path] = {}
    for rel_path, source in module_sources.items():
        abs_path = repo_root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(source + "\n", encoding="utf-8")
        written_paths[rel_path] = abs_path

    functions = _compute_function_meta(written_paths)
    goids = {
        "func_a": GOID_FUNC_A,
        "func_b": GOID_FUNC_B,
        "func_c": GOID_FUNC_C,
        "helper": GOID_HELPER,
    }
    func_meta: dict[str, CanonicalFunction] = {}
    for qualname, meta in functions.items():
        goid = goids.get(qualname)
        if goid is None:
            continue
        func_meta[qualname] = CanonicalFunction(
            qualname=qualname,
            goid=goid,
            rel_path=meta["rel_path"],
            start_line=meta["start_line"],
            end_line=meta["end_line"],
        )

    module_map = {
        MOD_A_FQN: MOD_A_PATH,
        MOD_B_FQN: MOD_B_PATH,
        MOD_C_FQN: MOD_C_PATH,
        MOD_UTIL_FQN: MOD_UTIL_PATH,
    }
    return CanonicalRepo(
        module_paths=module_map,
        goids=goids,
        functions=func_meta,
    )


def _compute_function_meta(written_paths: dict[str, Path]) -> dict[str, dict[str, int]]:
    """Parse written files to compute start/end lines."""
    meta: dict[str, dict[str, int]] = {}
    for rel_path, abs_path in written_paths.items():
        source = abs_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                qualname = node.name
                start_line = getattr(node, "lineno", 1)
                end_line = getattr(node, "end_lineno", start_line)
                meta[qualname] = {
                    "rel_path": rel_path,
                    "start_line": start_line,
                    "end_line": end_line,
                }
    return meta


__all__ = [
    "CanonicalFunction",
    "CanonicalRepo",
    "GOID_CALLEE",
    "GOID_CALLER",
    "GOID_FUNC_A",
    "GOID_FUNC_B",
    "GOID_FUNC_C",
    "GOID_HELPER",
    "MOD_A_FQN",
    "MOD_A_PATH",
    "MOD_B_FQN",
    "MOD_B_PATH",
    "MOD_C_FQN",
    "MOD_C_PATH",
    "MOD_UTIL_FQN",
    "MOD_UTIL_PATH",
    "write_canonical_repo",
]
