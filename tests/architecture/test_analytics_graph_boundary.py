"""Ensure analytics modules depend on graph engines rather than raw view loaders."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_analytics_modules_do_not_import_graph_views() -> None:
    """Guard against direct analytics -> graphs.engine.views coupling."""
    root = Path("src/codeintel/build/analytics")
    violations: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "codeintel.build.graphs.engine.views" in text:
            violations.append(str(path))
    if violations:
        pytest.fail(f"Replace graph views with GraphEngine in analytics modules: {violations}")
