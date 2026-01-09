"""Lint for raw pyarrow.compute usage in build/ingestion nodes."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_ALLOWLIST: frozenset[str] = frozenset(
    {
        "src/codeintel/build/tabular/arrow_ops.py",
        "src/codeintel/build/tabular/array_ops.py",
    }
)

_SCAN_DIRS: tuple[str, ...] = (
    "src/codeintel/build",
    "src/codeintel/ingestion",
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


def _is_pyarrow_compute_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(
            alias.name == "pyarrow.compute" or alias.name.startswith("pyarrow.compute.")
            for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == "pyarrow":
            return any(alias.name == "compute" for alias in node.names)
        return module == "pyarrow.compute" or module.startswith("pyarrow.compute.")
    return False


class _ComputeImportVisitor(ast.NodeVisitor):
    def __init__(self, *, path: Path) -> None:
        self.path = path
        self.violations: list[Violation] = []

    def visit_Import(self, node: ast.Import) -> None:
        if _is_pyarrow_compute_import(node):
            self.violations.append(
                Violation(
                    path=self.path,
                    lineno=node.lineno,
                    message=(
                        "Raw pyarrow.compute import detected; use core DSL helpers "
                        "(codeintel.core.columnar.expr_vocab/kernels) instead."
                    ),
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if _is_pyarrow_compute_import(node):
            self.violations.append(
                Violation(
                    path=self.path,
                    lineno=node.lineno,
                    message=(
                        "Raw pyarrow.compute import detected; use core DSL helpers "
                        "(codeintel.core.columnar.expr_vocab/kernels) instead."
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
    visitor = _ComputeImportVisitor(path=path)
    visitor.visit(tree)
    return visitor.violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pyarrow.compute import lint.

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
    output_lines.append(f"{len(violations)} raw pyarrow.compute import(s) detected.")
    sys.stderr.write("\n".join(output_lines) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
