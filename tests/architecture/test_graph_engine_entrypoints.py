"""Guard rails for graph engine construction entrypoints."""

from __future__ import annotations

from pathlib import Path

import pytest

ALLOWED = {
    Path("src/codeintel/graphs/engine_factory.py"),
    Path("src/codeintel/analytics/graph_runtime.py"),
}


def test_build_graph_engine_entrypoints_are_restricted() -> None:
    """Ensure build_graph_engine is only referenced in sanctioned modules."""
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src" / "codeintel"
    offenders: list[Path] = []
    for path in src_root.rglob("*.py"):
        relative = path.relative_to(repo_root)
        if relative in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        if "build_graph_engine(" in text:
            offenders.append(relative)
    if offenders:
        pytest.fail(f"Unexpected build_graph_engine usage in: {sorted(map(str, offenders))}")
