"""Guardrails for analytics rowset builders (ordering + list decoding helpers)."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_SCAN_ROOTS: tuple[str, ...] = ("src/codeintel/build/analytics",)
_ALLOWLIST_NO_DECODER: frozenset[str] = frozenset(
    {
        "src/codeintel/build/analytics/cfg_dfg/helpers.py",
    }
)
_DECODER_PREFIXES: tuple[str, ...] = ("_list_values", "_flatten_", "_normalize_")


@dataclass(frozen=True)
class Violation:
    """Single lint violation discovered during scanning."""

    path: Path
    lineno: int
    message: str


def _iter_python_files(root: Path) -> Iterable[Path]:
    for rel_root in _SCAN_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def _is_allowlisted(path: Path, *, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()
    return rel in _ALLOWLIST_NO_DECODER


def _contains_list_literal(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and child.value == "list":
            return True
    return False


def _is_list_aggregate_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != "aggregate":
        return False
    for arg in node.args:
        if _contains_list_literal(arg):
            return True
    for keyword in node.keywords:
        if keyword.arg == "aggregates" and _contains_list_literal(keyword.value):
            return True
    return False


def _is_order_by_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "order_by"


def _has_decoder_helper(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for prefix in _DECODER_PREFIXES:
                if node.name.startswith(prefix):
                    return True
    return False


def _lint_function(node: ast.FunctionDef, *, path: Path) -> list[Violation]:
    calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)]
    has_list_aggregate = any(_is_list_aggregate_call(call) for call in calls)
    has_order_by = any(_is_order_by_call(call) for call in calls)
    if has_list_aggregate and not has_order_by:
        return [
            Violation(
                path=path,
                lineno=node.lineno,
                message="Plan.aggregate(list) without order_by in rowset helper.",
            )
        ]
    return []


def _lint_file(path: Path, *, root: Path) -> list[Violation]:
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
    has_list_aggregate = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            violations.extend(_lint_function(node, path=path))
            calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)]
            if any(_is_list_aggregate_call(call) for call in calls):
                has_list_aggregate = True
    if (
        has_list_aggregate
        and not _is_allowlisted(path, root=root)
        and not _has_decoder_helper(tree)
    ):
        violations.append(
            Violation(
                path=path,
                lineno=1,
                message="List-aggregate rowset missing list-decoding helper.",
            )
        )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run analytics rowset guardrails.

    Parameters
    ----------
    argv
        Optional CLI args, with the repo root as the first entry.

    Returns
    -------
    int
        Exit code (0 for success, 1 for violations).
    """
    args = list(argv) if argv is not None else []
    root = Path(args[0]).resolve() if args else Path.cwd().resolve()
    violations: list[Violation] = []
    for path in _iter_python_files(root):
        violations.extend(_lint_file(path, root=root))

    if not violations:
        return 0

    output_lines = [
        f"{violation.path.relative_to(root)}:{violation.lineno}: {violation.message}"
        for violation in violations
    ]
    output_lines.append(f"{len(violations)} analytics rowset guardrail violation(s).")
    sys.stderr.write("\n".join(output_lines) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
