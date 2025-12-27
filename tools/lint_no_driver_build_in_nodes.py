"""Lint for runtime/driver construction inside Hamilton node modules."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_ALLOWLIST: frozenset[str] = frozenset()

_SCAN_DIRS: tuple[str, ...] = (
    "src/codeintel/build/hamilton/native",
    "src/codeintel/build/hamilton/nodes",
)

_BANNED_MODULE_PREFIXES: tuple[str, ...] = (
    "codeintel.runtime",
    "codeintel.serving.runtime",
    "codeintel.core.runtime.loader",
)

_BANNED_SYMBOLS: frozenset[str] = frozenset(
    {
        "RuntimeBundle",
        "build_driver",
        "build_runtime",
        "build_runtime_primitives",
        "compose_runtime",
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


def _is_allowlisted(path: Path, *, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()
    return rel in _ALLOWLIST


def _is_banned_module(module: str) -> bool:
    return module in _BANNED_MODULE_PREFIXES or module.startswith(_BANNED_MODULE_PREFIXES)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _DriverBuildVisitor(ast.NodeVisitor):
    def __init__(self, *, path: Path) -> None:
        self.path = path
        self.violations: list[Violation] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if _is_banned_module(alias.name):
                self.violations.append(
                    Violation(
                        path=self.path,
                        lineno=node.lineno,
                        message=(
                            "Runtime/driver modules are not allowed in DAG nodes; "
                            f"found import '{alias.name}'."
                        ),
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module and _is_banned_module(module):
            self.violations.append(
                Violation(
                    path=self.path,
                    lineno=node.lineno,
                    message=(
                        "Runtime/driver modules are not allowed in DAG nodes; "
                        f"found import from '{module}'."
                    ),
                )
            )
        for alias in node.names:
            if alias.name in _BANNED_SYMBOLS:
                self.violations.append(
                    Violation(
                        path=self.path,
                        lineno=node.lineno,
                        message=(
                            "Runtime/driver construction is not allowed in DAG nodes; "
                            f"found import '{alias.name}'."
                        ),
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in _BANNED_SYMBOLS:
            self.violations.append(
                Violation(
                    path=self.path,
                    lineno=node.lineno,
                    message=(
                        "Runtime/driver construction is not allowed in DAG nodes; "
                        f"found call to '{name}'."
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
    visitor = _DriverBuildVisitor(path=path)
    visitor.visit(tree)
    return visitor.violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the runtime/driver build lint.

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
    output_lines.append(f"{len(violations)} runtime/driver build(s) detected.")
    sys.stderr.write("\n".join(output_lines) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
