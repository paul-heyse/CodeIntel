"""Ensure data model consumers do not rely on legacy JSON blobs."""

from __future__ import annotations

from pathlib import Path

import pytest

TOKENS: tuple[str, ...] = ("fields_json", "relationships_json")
ALLOWED: set[str] = {
    "src/codeintel/analytics/data_models.py",
    "src/codeintel/config/schemas/tables.py",
    "src/codeintel/storage/data_models.py",
    "src/codeintel/storage/views.py",
}


def test_no_legacy_json_usage_outside_whitelist() -> None:
    """Fail when legacy JSON columns are referenced outside whitelisted modules."""
    repo_root = Path(__file__).resolve().parents[2]
    violations: list[str] = []
    for path in repo_root.joinpath("src").rglob("*.py"):
        rel = path.relative_to(repo_root).as_posix()
        if rel in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        for token in TOKENS:
            if token not in text:
                continue
            lines = [
                str(idx) for idx, line in enumerate(text.splitlines(), start=1) if token in line
            ]
            violations.append(f"{rel}:{token}@{','.join(lines)}")
    if violations:
        detail = ", ".join(sorted(violations))
        pytest.fail(f"Legacy data model JSON reference found: {detail}")
