"""Guardrails enforcing the testing charter (no patching, realistic wiring)."""

from __future__ import annotations

import re
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
SELF_PATH = TEST_ROOT / "test_testing_contract.py"


FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "monkeypatch": re.compile(r"\bmonkeypatch\b"),
    "pytest-mock": re.compile(r"pytest-mock"),
    "unittest.mock": re.compile(r"\bunittest\.mock\b"),
    "patch-decorator": re.compile(r"@patch\b"),
    "sys.path-mutation": re.compile(r"sys\.path"),
    "pythonpath-edit": re.compile(r"PYTHONPATH"),
    "private-import": re.compile(r"from\s+codeintel[.\w]*\s+import\s+_"),
}


def _iter_test_files() -> list[Path]:
    return [path for path in TEST_ROOT.rglob("*.py") if path != SELF_PATH]


def test_testing_charter_forbidden_patterns() -> None:
    """
    Fail fast when tests use forbidden patching or path hacks.

    Enforces the charter: no monkeypatching/runtime patching, no sys.path edits,
    and no imports of underscore-prefixed production symbols from tests.

    Raises
    ------
    AssertionError
        If any forbidden pattern is detected in the test suite.
    """
    violations: list[str] = []
    for path in _iter_test_files():
        content = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(content):
                violations.append(f"{label}: {path}")
    if violations:
        message = "Testing charter violations detected:\n" + "\n".join(sorted(violations))
        raise AssertionError(message)
