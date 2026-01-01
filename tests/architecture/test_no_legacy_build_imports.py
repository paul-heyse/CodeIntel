"""Guard against new legacy analytics/graphs imports outside allowed packages."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "codeintel"
ALLOWED_PREFIXES = (
    "src/codeintel/build/analytics/",
    "src/codeintel/build/graphs/",
    "src/codeintel/build/hamilton/native/",
)


def test_no_legacy_build_imports_outside_allowed() -> None:
    """Prevent new code from importing legacy analytics/graphs packages."""
    needles = ("codeintel.build.analytics", "codeintel.build.graphs")
    violations: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        if rel.startswith(ALLOWED_PREFIXES):
            continue
        text = path.read_text(encoding="utf-8")
        if any(needle in text for needle in needles):
            violations.append(rel)
    if violations:
        message = "Legacy build imports outside allowed packages:\n" + "\n".join(sorted(violations))
        pytest.fail(message)
