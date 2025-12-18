"""Fast invariant scans over `src/codeintel/build/hamilton/native`."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def _iter_build_native_files() -> list[Path]:
    """Return all Python files under `src/codeintel/build/hamilton/native`.

    Returns
    -------
    list[Path]
        Sorted list of file paths.
    """
    root = Path(__file__).resolve().parents[3]
    native_dir = root / "src" / "codeintel" / "build" / "hamilton" / "native"
    return sorted(native_dir.rglob("*.py"))


def test_build_native_modules_do_not_call_gateway_policy_writes() -> None:
    """Native DAG modules must not call storage policy write APIs directly."""
    pattern = re.compile(
        r"\.policy\.(?:delete_for_snapshot|bulk_insert_mappings|bulk_insert|delete)\("
    )
    for path in _iter_build_native_files():
        text = path.read_text(encoding="utf-8")
        match = pattern.search(text)
        if match is None:
            continue
        line = text.count("\n", 0, match.start()) + 1
        pytest.fail(f"Forbidden gateway.policy write call found: {path}:{line}")


def test_build_native_modules_do_not_use_direct_ibis_table_accessor() -> None:
    """Native DAG modules must not use `gateway.ibis.table(...)` directly."""
    pattern = re.compile(r"\.ibis\.table\(")
    for path in _iter_build_native_files():
        text = path.read_text(encoding="utf-8")
        match = pattern.search(text)
        if match is None:
            continue
        line = text.count("\n", 0, match.start()) + 1
        pytest.fail(f"Forbidden `.ibis.table(...)` usage found: {path}:{line}")
