"""Ensure compute nodes avoid direct env.gateway access."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ALLOWED_DECORATORS = {
    "tag_helper",
    "tag_loader_query",
    "tag_loader_dataframe",
}
_CHECK_DECORATORS = {
    "codeintel_target",
    "tag_artifact",
    "tag_compute",
    "tag_dataset",
    "tag_materialize",
    "tag_tool",
}


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _has_env_gateway_access(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Attribute)
            and child.attr == "gateway"
            and isinstance(child.value, ast.Name)
            and child.value.id == "env"
        ):
            return True
    return False


def _should_check(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    decorator_names = {_decorator_name(decorator) for decorator in func.decorator_list}
    if any(name in _ALLOWED_DECORATORS for name in decorator_names if name is not None):
        return False
    return any(name in _CHECK_DECORATORS for name in decorator_names if name is not None)


def _scan_module(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _should_check(node):
            continue
        if _has_env_gateway_access(node):
            violations.append(f"{path.name}:{node.name}")
    return violations


def test_no_env_gateway_in_compute_nodes() -> None:
    """Ensure compute nodes do not access env.gateway directly."""
    repo_root = Path(__file__).resolve().parents[3]
    base = repo_root / "src" / "codeintel" / "build" / "hamilton" / "native"
    paths = [base / "analytics", base / "graphs"]

    violations: list[str] = []
    for directory in paths:
        for path in directory.rglob("*.py"):
            if path.name == "__init__.py":
                continue
            violations.extend(_scan_module(path))

    if violations:
        joined = ", ".join(sorted(violations))
        pytest.fail(f"env.gateway access found in compute nodes: {joined}")
