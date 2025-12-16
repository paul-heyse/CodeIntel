"""PR-52: Verify no legacy orchestrators remain outside build system."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src" / "codeintel"


def _iter_py_files(root: Path) -> list[Path]:
    """Return Python files under a directory tree.

    Parameters
    ----------
    root
        Root directory to scan.

    Returns
    -------
    list[pathlib.Path]
        Python files under ``root`` excluding ``__pycache__`` paths.
    """
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def _relative_path(path: Path) -> str:
    """Return a stable forward-slashed path relative to repo root.

    Parameters
    ----------
    path
        File path under the repository.

    Returns
    -------
    str
        Repository-relative path with forward slashes.
    """
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def _contains_ibis_write_call(source: str) -> bool:
    """Return True when source contains an `.ibis.write(...)` call.

    Parameters
    ----------
    source
        Python source text.

    Returns
    -------
    bool
        True when an `.ibis.write(...)` call is present.
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


def test_pr52_no_direct_writes_outside_build() -> None:
    """Verify no `.ibis.write()` calls exist outside build/storage layers."""
    allow_prefixes = ("src/codeintel/build/", "src/codeintel/storage/")
    bad: list[str] = []
    for path in _iter_py_files(SRC_ROOT):
        rel = _relative_path(path)
        if rel.startswith(allow_prefixes):
            continue
        if _contains_ibis_write_call(path.read_text(encoding="utf-8")):
            bad.append(rel)
    if bad:
        message = "Direct DB writes outside build/storage:\n" + "\n".join(sorted(bad))
        pytest.fail(message)


def test_pr52_no_deprecated_function_calls_in_cli() -> None:
    """Verify CLI handlers do not call deleted compute/build helper functions."""
    deprecated_patterns = (
        "compute_cfg_metrics(",
        "compute_dfg_metrics(",
        "compute_data_models(",
        "compute_function_history(",
        "compute_history_timeseries(",
        "compute_test_graph_metrics(",
        "build_entrypoints(",
        "build_external_dependency_calls(",
        "build_external_dependencies(",
    )
    cli_root = SRC_ROOT / "cli"
    bad: list[tuple[str, str]] = []
    for path in _iter_py_files(cli_root):
        text = path.read_text(encoding="utf-8")
        bad.extend(
            (_relative_path(path), pattern) for pattern in deprecated_patterns if pattern in text
        )
    if bad:
        message = "Deprecated calls in CLI:\n" + "\n".join(f"{p}: {pat}" for p, pat in bad)
        pytest.fail(message)


def test_pr52_empty_plugin_directories_removed() -> None:
    """Verify removed analytics plugin directories do not exist."""
    plugins_root = SRC_ROOT / "build" / "plugins" / "analytics"
    should_not_exist = (
        "cfg_dfg",
        "data_models",
        "dependencies",
        "entrypoints",
        "history",
    )
    bad: list[str] = []
    for name in should_not_exist:
        path = plugins_root / name
        if path.exists():
            bad.append(_relative_path(path))
    if bad:
        message = "Empty plugin directories should be deleted:\n" + "\n".join(sorted(bad))
        pytest.fail(message)


def test_pr52_migrated_plugins_exist_as_native_modules() -> None:
    """Verify previously non-migrated plugins are now native Hamilton modules.

    These plugins were migrated in Phase 4 of the Hamilton Native Implementation Plan.
    """
    native_root = SRC_ROOT / "build" / "hamilton" / "native" / "analytics"
    should_exist = (
        "config_data_flow.py",
        "coverage_test_edges.py",
    )
    missing: list[str] = []
    for file_name in should_exist:
        path = native_root / file_name
        if not path.exists():
            missing.append(_relative_path(path))
    if missing:
        message = "Native Hamilton modules missing:\n" + "\n".join(sorted(missing))
        pytest.fail(message)
