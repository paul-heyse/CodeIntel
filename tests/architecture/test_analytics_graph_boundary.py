"""Ensure analytics modules depend on graph engines rather than NetworkX views."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_analytics_modules_do_not_import_nx_views() -> None:
    """Guard against direct analytics -> graphs.nx_views coupling."""
    root = Path("src/codeintel/analytics")
    violations: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "codeintel.graphs.nx_views" in text:
            violations.append(str(path))
    if violations:
        pytest.fail(f"Replace nx_views with GraphEngine in analytics modules: {violations}")
