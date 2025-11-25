"""Ensure subprocess usage is centralized in tool runner/service modules."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_no_direct_subprocess_usage_outside_tooling() -> None:
    """Fail when subprocess usage appears outside the centralized tooling modules."""
    repo_root = Path().resolve()
    src_root = repo_root / "src" / "codeintel"
    allowed = {
        Path("src/codeintel/ingestion/tool_runner.py"),
        Path("src/codeintel/ingestion/tool_service.py"),
    }
    patterns = ("create_subprocess_exec(", "subprocess.run(", "Popen(")

    violations: list[str] = []
    for path in src_root.rglob("*.py"):
        rel_path = path.relative_to(repo_root)
        if rel_path in allowed:
            continue
        content = path.read_text(encoding="utf8")
        if any(pattern in content for pattern in patterns):
            violations.append(str(rel_path))

    if violations:
        pytest.fail(f"Direct subprocess usage found in: {', '.join(sorted(violations))}")
