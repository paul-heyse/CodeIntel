"""Guardrails for nx_views usage outside the graphs package."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_nx_views_only_used_in_graphs_layer() -> None:
    """Ensure nx_views imports stay within codeintel.graphs.* modules."""
    root = Path("src/codeintel")
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if "graphs" in path.parts and path.parts[path.parts.index("codeintel") + 1] == "graphs":
            continue
        text = path.read_text(encoding="utf-8")
        if "codeintel.graphs.nx_views" in text:
            violations.append(str(path))
    if violations:
        pytest.fail(f"nx_views imports should be confined to graphs package: {violations}")
