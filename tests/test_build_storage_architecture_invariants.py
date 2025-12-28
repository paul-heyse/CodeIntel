"""Architecture invariants for the build/storage data operations boundary."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "codeintel"

_IBIS_TABLE_PATTERN = re.compile(r"\.ibis\.table\(")
_POLICY_WRITE_PATTERN = re.compile(
    r"\.policy\.(?:delete_for_snapshot|bulk_insert_mappings|bulk_insert|delete)\("
)

_ALLOWLIST_IBIS_TABLE_FILES = {
    "src/codeintel/build/hamilton/validate.py",
}


def _iter_source_files() -> list[Path]:
    return [path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in str(path)]


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def _line_numbers(text: str, pattern: re.Pattern[str]) -> list[int]:
    return [text.count("\n", 0, match.start()) + 1 for match in pattern.finditer(text)]


def test_no_direct_ibis_table_calls_outside_allowlist() -> None:
    """Assert `.ibis.table(...)` only appears in explicitly allowlisted files."""
    offenders: list[str] = []
    for path in _iter_source_files():
        rel = _rel(path)
        if rel in _ALLOWLIST_IBIS_TABLE_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        lines = _line_numbers(text, _IBIS_TABLE_PATTERN)
        offenders.extend([f"{rel}:{line}" for line in lines])
    if offenders:
        message = "Direct `.ibis.table(...)` calls found:\n" + "\n".join(sorted(offenders))
        pytest.fail(message)


def test_no_build_policy_write_calls() -> None:
    """Assert build modules do not call snapshot-write policy APIs directly."""
    offenders: list[str] = []
    build_root = SRC_ROOT / "build"
    for path in build_root.rglob("*.py"):
        rel = _rel(path)
        text = path.read_text(encoding="utf-8")
        lines = _line_numbers(text, _POLICY_WRITE_PATTERN)
        offenders.extend([f"{rel}:{line}" for line in lines])
    if offenders:
        message = "Build policy write calls found:\n" + "\n".join(sorted(offenders))
        pytest.fail(message)
