"""Ensure core package stays dependency-light (stdlib + config primitives)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ALLOWED_PREFIXES = {
    "typing",
    "collections",
    "pathlib",
    "dataclasses",
    "itertools",
    "types",
    "sys",
}


def _is_allowed_module(module: str) -> bool:
    base = module.split(".", 1)[0]
    return base in ALLOWED_PREFIXES or base in sys.stdlib_module_names


def _collect_bad_imports(core_root: Path) -> list[tuple[str, str]]:
    bad_imports: list[tuple[str, str]] = []

    def _record(module_name: str, rel: str, prefix: str) -> None:
        if module_name in {"__future__", ""}:
            return
        if module_name.startswith("codeintel.config.primitives"):
            return
        if _is_allowed_module(module_name):
            return
        bad_imports.append((rel, f"{prefix} {module_name}"))

    for path in core_root.rglob("*.py"):
        rel = path.relative_to(core_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                _record(module, rel, "from")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    _record(alias.name, rel, "import")
    return bad_imports


def test_core_imports_are_stdlib_only() -> None:
    """Core must only import stdlib or codeintel.config.primitives."""
    core_root = Path(__file__).resolve().parent / "src" / "codeintel" / "core"
    bad_imports = _collect_bad_imports(core_root)
    if bad_imports:
        formatted = "; ".join(f"{path}: {msg}" for path, msg in bad_imports)
        pytest.fail(f"Non-stdlib imports in core: {formatted}")
