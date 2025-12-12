"""Layering guardrails preventing forbidden imports into middle packages."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ALLOWED_SERVING_IMPORTERS = {"cli", "core", "export", "serving", "tests"}


def _iter_python_files(root: Path) -> list[Path]:
    return list(root.rglob("*.py")) + list(root.rglob("*.pyi"))


def _is_type_checking_guard(node: ast.If) -> bool:
    """Check if an If node is a TYPE_CHECKING guard.

    Parameters
    ----------
    node
        AST If node to check.

    Returns
    -------
    bool
        True if the test is `TYPE_CHECKING` or `typing.TYPE_CHECKING`.
    """
    test = node.test

    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True

    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
    )


def _collect_type_checking_imports(tree: ast.Module) -> set[int]:
    """Collect line numbers of imports inside TYPE_CHECKING blocks.

    Parameters
    ----------
    tree
        Parsed AST module.

    Returns
    -------
    set[int]
        Set of line numbers for imports inside TYPE_CHECKING guards.
    """
    type_checking_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    type_checking_lines.add(child.lineno)
    return type_checking_lines


def _assert_no_imports(
    package_root: Path,
    forbidden_prefix: str,
    allowed_top_levels: set[str],
) -> None:
    """Assert no forbidden imports exist outside TYPE_CHECKING blocks.

    Parameters
    ----------
    package_root
        Root directory of the package to scan.
    forbidden_prefix
        Import prefix to forbid (e.g., "codeintel.serving").
    allowed_top_levels
        Set of top-level package names that may use forbidden imports.
    """
    bad_imports: list[tuple[str, str]] = []
    for py_path in _iter_python_files(package_root):
        rel = py_path.relative_to(package_root)
        top_level = rel.parts[0]
        if top_level in allowed_top_levels:
            continue

        tree = ast.parse(py_path.read_text(encoding="utf-8"))
        type_checking_lines = _collect_type_checking_imports(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.lineno in type_checking_lines:
                    continue
                module = node.module or ""
                if module.startswith(forbidden_prefix):
                    bad_imports.append((rel.as_posix(), f"from {module} import ..."))
            elif isinstance(node, ast.Import):
                if node.lineno in type_checking_lines:
                    continue
                bad_imports.extend(
                    (rel.as_posix(), f"import {alias.name}")
                    for alias in node.names
                    if alias.name.startswith(forbidden_prefix)
                )

    if bad_imports:
        formatted = "; ".join(f"{path}: {message}" for path, message in bad_imports)
        pytest.fail(f"Disallowed imports of {forbidden_prefix}: {formatted}")


def test_no_serving_imports_in_middle_packages() -> None:
    """Ensure analytics/graphs/ingestion/storage do not import codeintel.serving.*.

    TYPE_CHECKING imports are allowed since they don't create runtime dependencies.
    """
    package_root = Path(__file__).resolve().parent.parent / "src" / "codeintel"
    _assert_no_imports(
        package_root=package_root,
        forbidden_prefix="codeintel.serving",
        allowed_top_levels=ALLOWED_SERVING_IMPORTERS,
    )
