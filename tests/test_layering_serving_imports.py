"""Layering guardrails preventing forbidden imports into middle packages."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ALLOWED_SERVING_IMPORTERS = {"cli", "pipeline", "serving", "tests"}
ALLOWED_PIPELINE_IMPORTERS = {"pipeline", "serving", "cli", "tests", "storage"}


def _iter_python_files(root: Path) -> list[Path]:
    return list(root.rglob("*.py")) + list(root.rglob("*.pyi"))


def _assert_no_imports(
    package_root: Path,
    forbidden_prefix: str,
    allowed_top_levels: set[str],
) -> None:
    bad_imports: list[tuple[str, str]] = []
    for py_path in _iter_python_files(package_root):
        rel = py_path.relative_to(package_root)
        top_level = rel.parts[0]
        if top_level in allowed_top_levels:
            continue

        tree = ast.parse(py_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(forbidden_prefix):
                    bad_imports.append((rel.as_posix(), f"from {module} import ..."))
            elif isinstance(node, ast.Import):
                bad_imports.extend(
                    (rel.as_posix(), f"import {alias.name}")
                    for alias in node.names
                    if alias.name.startswith(forbidden_prefix)
                )

    if bad_imports:
        formatted = "; ".join(f"{path}: {message}" for path, message in bad_imports)
        pytest.fail(f"Disallowed imports of {forbidden_prefix}: {formatted}")


def test_no_serving_imports_in_middle_packages() -> None:
    """Ensure analytics/graphs/ingestion/storage do not import codeintel.serving.*."""
    package_root = Path(__file__).resolve().parent.parent / "src" / "codeintel"
    _assert_no_imports(
        package_root=package_root,
        forbidden_prefix="codeintel.serving",
        allowed_top_levels=ALLOWED_SERVING_IMPORTERS,
    )


def test_no_pipeline_imports_in_middle_packages() -> None:
    """Ensure analytics/graphs/ingestion/storage do not import codeintel.pipeline.*."""
    package_root = Path(__file__).resolve().parent.parent / "src" / "codeintel"
    _assert_no_imports(
        package_root=package_root,
        forbidden_prefix="codeintel.pipeline",
        allowed_top_levels=ALLOWED_PIPELINE_IMPORTERS,
    )
