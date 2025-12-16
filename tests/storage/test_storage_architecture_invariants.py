"""Architecture guardrails for the Hamilton-first storage layer."""

from __future__ import annotations

import re
from pathlib import Path

from tests._helpers.assertions import expect_false, expect_not_in


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_storage_has_no_build_imports() -> None:
    """Storage must not import build at runtime."""
    root = _repo_root()
    storage_root = root / "src" / "codeintel" / "storage"
    pattern = re.compile(r"\b(from|import)\s+codeintel\.build\b")

    offenders: list[str] = []
    for path in storage_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path.relative_to(root)))

    expect_false(
        offenders,
        message=f"Storage must not import codeintel.build; offenders: {offenders}",
    )


def test_storage_legacy_modules_removed() -> None:
    """Legacy compatibility modules should be fully decommissioned."""
    root = _repo_root()
    removed = [
        "src/codeintel/storage/build_bridge.py",
        "src/codeintel/storage/gateway_cache.py",
        "src/codeintel/storage/validation/data_checks.py",
        "src/codeintel/storage/gateway/insert_helpers.py",
        "src/codeintel/storage/queries/execution.py",
        "src/codeintel/storage/helpers/profiling.py",
        "src/codeintel/storage/view_names.py",
        "src/codeintel/storage/views/creation.py",
        "src/codeintel/storage/datasets/scaffold.py",
    ]
    present = [path for path in removed if (root / path).exists()]
    expect_false(
        present,
        message=f"Legacy storage modules should be deleted; found: {present}",
    )


def test_ibis_gateway_has_no_sql_escape_hatch() -> None:
    """IbisGateway.sql should not exist (raw SQL must stay internal)."""
    root = _repo_root()
    path = root / "src" / "codeintel" / "storage" / "ibis_adapter.py"
    text = path.read_text(encoding="utf-8")
    expect_not_in("def sql(self, raw_sql", text)


def test_no_gateway_accessor_inserts_in_src() -> None:
    """The typed gateway accessors are read-only; insert_* calls are disallowed in src."""
    root = _repo_root()
    code_root = root / "src" / "codeintel"
    pattern = re.compile(r"\.(core|graph|analytics)\.insert_[A-Za-z0-9_]+\(")

    offenders: list[str] = []
    for path in code_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path.relative_to(root)))

    expect_false(
        offenders,
        message=f"Disallowed gateway accessor insert_* calls in src: {offenders}",
    )
