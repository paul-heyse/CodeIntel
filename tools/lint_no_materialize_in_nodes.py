"""Lint for table materialization in build/ingestion nodes."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_ALLOWLIST: frozenset[str] = frozenset(
    {
        "src/codeintel/ingestion/compute/cst_extract.py",
        "src/codeintel/ingestion/compute/tree_sitter_index.py",
    }
)

_SCAN_DIRS: tuple[str, ...] = (
    "src/codeintel/build/hamilton",
    "src/codeintel/ingestion/compute",
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


def _is_allowlisted(path: Path, *, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()
    return rel in _ALLOWLIST


class _MaterializeVisitor(ast.NodeVisitor):
    def __init__(self, *, path: Path) -> None:
        self.path = path
        self.violations: list[Violation] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to_table":
            self.violations.append(
                Violation(
                    path=self.path,
                    lineno=node.lineno,
                    message=(
                        "Materialization via to_table() detected; keep readers streaming "
                        "and finalize at explicit boundaries."
                    ),
                )
            )
        self.generic_visit(node)


def _lint_file(path: Path, *, root: Path) -> list[Violation]:
    if _is_allowlisted(path, root=root):
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _MaterializeVisitor(path=path)
    visitor.visit(tree)
    return visitor.violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the materialization lint.

    Parameters
    ----------
    argv
        Optional CLI arguments, with the repo root as the first entry.

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
    output_lines.append(f"{len(violations)} materialization call(s) detected.")
    sys.stderr.write("\n".join(output_lines) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
