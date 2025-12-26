"""Storage boundary tests for graph telemetry inputs."""

from __future__ import annotations

from pathlib import Path

_GRAPH_MODULES: tuple[Path, ...] = (
    Path("src/codeintel/build/hamilton/native/graphs/call_graph.py"),
    Path("src/codeintel/build/hamilton/native/graphs/cfg_dfg.py"),
    Path("src/codeintel/build/hamilton/native/graphs/graph_targets.py"),
    Path("src/codeintel/build/hamilton/native/graphs/import_graph.py"),
)


def test_graph_modules_avoid_direct_ibis_facade_reads() -> None:
    """Graph modules should not use ibis_facade.table directly."""
    for path in _GRAPH_MODULES:
        text = path.read_text(encoding="utf-8")
        assert "ibis_facade.table" not in text
