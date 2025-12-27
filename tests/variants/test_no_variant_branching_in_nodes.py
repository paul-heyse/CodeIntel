"""Ensure native Hamilton nodes avoid runtime variant branching."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Violation:
    path: Path
    lineno: int
    message: str


def _iter_native_modules(root: Path) -> list[Path]:
    native_root = root / "src" / "codeintel" / "build" / "hamilton" / "native"
    if not native_root.exists():
        return []
    return [path for path in native_root.rglob("*.py") if "__pycache__" not in path.parts]


def _find_variant_accesses(path: Path) -> list[_Violation]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: list[_Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Attribute):
                continue
            if isinstance(child.value, ast.Name) and child.value.id == "cfg":
                violations.append(
                    _Violation(
                        path=path,
                        lineno=child.lineno,
                        message="cfg.<...> access in node bodies is disallowed",
                    )
                )
            if (
                isinstance(child.value, ast.Name)
                and child.value.id == "env"
                and child.attr in {"config", "variants"}
            ):
                violations.append(
                    _Violation(
                        path=path,
                        lineno=child.lineno,
                        message=f"env.{child.attr} access in node bodies is disallowed",
                    )
                )
    return violations


def test_no_variant_branching_in_nodes() -> None:
    """Reject variant branching inside native DAG node bodies.

    Raises
    ------
    AssertionError
        If forbidden variant access is detected.
    """
    root = Path(__file__).resolve().parents[2]
    violations: list[_Violation] = []
    for path in _iter_native_modules(root):
        violations.extend(_find_variant_accesses(path))

    if violations:
        rendered = "\n".join(
            f"{violation.path.relative_to(root)}:{violation.lineno}: {violation.message}"
            for violation in violations
        )
        msg = "Variant branching detected in native nodes:\n" + rendered
        raise AssertionError(msg)
