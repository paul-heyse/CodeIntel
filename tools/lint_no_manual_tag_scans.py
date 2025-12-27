"""Lint for manual Hamilton tag scans over dr.graph.nodes."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_ALLOWLIST: frozenset[str] = frozenset(
    {
        "src/codeintel/build/hamilton/dag_catalog_compiler.py",
        "src/codeintel/build/hamilton/validate.py",
    }
)

_SCAN_DIRS: tuple[str, ...] = ("src", "tests", "tools")


@dataclass(frozen=True)
class Violation:
    """Single lint violation discovered during scanning."""

    path: Path
    lineno: int
    message: str


def _iter_python_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix == ".py":
            yield root
        return
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


def _is_graph_nodes_access(node: ast.AST) -> bool:
    if not isinstance(node, ast.Attribute):
        return False
    if node.attr != "nodes":
        return False
    value = node.value
    return isinstance(value, ast.Attribute) and value.attr == "graph"


def _is_graph_nodes_iter(node: ast.AST) -> bool:
    if _is_graph_nodes_access(node):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"items", "values", "keys"} and _is_graph_nodes_access(node.func.value):
            return True
    return False


def _bound_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            names.update(_bound_names(elt))
        return names
    return set()


class _TagAccessVisitor(ast.NodeVisitor):
    def __init__(self, bound_names: set[str]) -> None:
        self.bound_names = bound_names
        self.found = False

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.attr == "tags":
            if node.value.id in self.bound_names:
                self.found = True
        self.generic_visit(node)


class _ManualTagScanVisitor(ast.NodeVisitor):
    def __init__(self, *, path: Path) -> None:
        self.path = path
        self.violations: list[Violation] = []

    def visit_For(self, node: ast.For) -> None:
        self._check_loop(node)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._check_loop(node)
        self.generic_visit(node)

    def _check_loop(self, node: ast.stmt) -> None:
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            return
        if not _is_graph_nodes_iter(node.iter):
            return
        bound_names = _bound_names(node.target)
        if not bound_names:
            return
        visitor = _TagAccessVisitor(bound_names)
        for stmt in node.body:
            visitor.visit(stmt)
            if visitor.found:
                self.violations.append(
                    Violation(
                        path=self.path,
                        lineno=node.lineno,
                        message=(
                            "Manual tag scan over .graph.nodes is disallowed; "
                            "use list_available_variables(tag_filter=...) or TagQuery."
                        ),
                    )
                )
                break


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
    visitor = _ManualTagScanVisitor(path=path)
    visitor.visit(tree)
    return visitor.violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the manual tag scan lint.

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

    for violation in violations:
        rel = violation.path.relative_to(root)
        print(f"{rel}:{violation.lineno}: {violation.message}")
    print(f"{len(violations)} manual tag scan(s) detected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
