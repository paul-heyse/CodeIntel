"""Guardrail: analytics write_dataset calls must finalize within the function."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_SCAN_DIRS: tuple[str, ...] = ("src",)
_FINALIZE_CALLS: frozenset[str] = frozenset(
    {
        "finalize_analytics_table",
        "finalize_analytics_result",
        "finalize_table",
        "_finalize_rows_for_parquet",
    }
)


@dataclass(frozen=True)
class Violation:
    """Single lint violation discovered during scanning."""

    path: Path
    lineno: int
    message: str


def _iter_python_files(root: Path) -> Iterable[Path]:
    for dirname in _SCAN_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _table_key_from_call(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg != "table_key":
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    return None


def _has_finalize_call(node: ast.FunctionDef) -> bool:
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        name = _call_name(call)
        if name in _FINALIZE_CALLS:
            return True
    return False


def _lint_function(node: ast.FunctionDef, *, path: Path) -> list[Violation]:
    violations: list[Violation] = []
    has_finalize = _has_finalize_call(node)
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if _call_name(call) != "write_dataset":
            continue
        table_key = _table_key_from_call(call)
        if table_key is None or not table_key.startswith("analytics."):
            continue
        if not has_finalize:
            violations.append(
                Violation(
                    path=path,
                    lineno=call.lineno,
                    message=(
                        "analytics write_dataset call without finalize_analytics_* in "
                        "the same function."
                    ),
                )
            )
    return violations


def _lint_file(path: Path) -> list[Violation]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    if not isinstance(tree, ast.Module):
        return []
    violations: list[Violation] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            violations.extend(_lint_function(node, path=path))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analytics finalize write guardrail.

    Returns
    -------
    int
        Exit code (0 for success, 1 for violations).
    """
    args = list(argv) if argv is not None else []
    root = Path(args[0]).resolve() if args else Path.cwd().resolve()
    violations: list[Violation] = []
    for path in _iter_python_files(root):
        violations.extend(_lint_file(path))

    if not violations:
        return 0

    output_lines = [
        f"{violation.path.relative_to(root)}:{violation.lineno}: {violation.message}"
        for violation in violations
    ]
    output_lines.append(f"{len(violations)} analytics finalize write violation(s).")
    sys.stderr.write("\n".join(output_lines) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
