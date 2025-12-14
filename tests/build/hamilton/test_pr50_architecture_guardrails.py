"""PR50: architecture guardrails enforcing Hamilton-first conventions."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src" / "codeintel"

# All deprecated functions with direct DB writes have been removed in Phase 3-4.
# Hamilton native modules in build/hamilton/native/analytics/ are now the canonical path.
ALLOWLIST_IBIS_WRITE_FILES: set[str] = set()


def _iter_py_files() -> list[Path]:
    """Return all Python source files under `src/codeintel`.

    Returns
    -------
    list[pathlib.Path]
        Python source files to scan (excluding `__pycache__` paths).
    """
    return [path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in str(path)]


def _relative_path(path: Path) -> str:
    """Return a stable forward-slashed path relative to repo root.

    Returns
    -------
    str
        Normalized path suitable for stable test output.
    """
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def _contains_ibis_write_call(source: str) -> bool:
    """Return True when source contains an `.ibis.write(...)` call.

    Returns
    -------
    bool
        True when an ibis write call is detected.
    """
    try:
        module = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "write":
            continue
        target = func.value
        if isinstance(target, ast.Attribute) and target.attr == "ibis":
            return True
    return False


def test_pr50_no_plugin_registry_imports() -> None:
    """Verify no source file references the removed plugin_registry module."""
    bad: list[str] = []
    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8")
        if "codeintel.build.plugin_registry" in text:
            bad.append(_relative_path(path))
    if bad:
        message = "plugin_registry imports still exist:\n" + "\n".join(bad)
        pytest.fail(message)


def test_pr50_no_analytics_runtime_imports() -> None:
    """Verify no source file references the removed analytics.runtime package."""
    bad: list[str] = []
    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8")
        if "codeintel.analytics.runtime" in text:
            bad.append(_relative_path(path))
    if bad:
        message = "analytics.runtime imports still exist:\n" + "\n".join(bad)
        pytest.fail(message)


def test_pr50_no_ibis_write_outside_build_allowlist() -> None:
    """Verify `.ibis.write(...)` calls are restricted to build or allowlisted legacy writers."""
    allow_prefixes = (
        "src/codeintel/build/",
        "src/codeintel/storage/",
    )
    offenders: list[str] = []
    for path in _iter_py_files():
        rel = _relative_path(path)
        if rel.startswith(allow_prefixes) or rel in ALLOWLIST_IBIS_WRITE_FILES:
            continue
        if _contains_ibis_write_call(path.read_text(encoding="utf-8")):
            offenders.append(rel)

    if offenders:
        message = "Direct DB writes found outside build/storage:\n" + "\n".join(offenders)
        pytest.fail(message)
