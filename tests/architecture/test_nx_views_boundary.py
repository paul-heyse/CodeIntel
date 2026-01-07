"""Guardrails for graph view loader usage outside the graphs package."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_graph_views_only_used_in_graphs_layer() -> None:
    """Ensure graph view imports stay within codeintel.build.graphs.* modules."""
    root = Path("src/codeintel")
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if "graphs" in path.parts and path.parts[path.parts.index("codeintel") + 1] == "graphs":
            continue
        text = path.read_text(encoding="utf-8")
        if "codeintel.build.graphs.engine.views" in text:
            violations.append(str(path))
    if violations:
        pytest.fail(f"graph views imports should be confined to graphs package: {violations}")
