"""Configuration dataclasses for graph test environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

    from codeintel.config import ConfigBuilder
    from codeintel.storage.gateway import StorageGateway


REPO = "demo/repo"
COMMIT = "deadbeef"


@dataclass(frozen=True)
class SpanSnapshot:
    """Collected GOID/symbol-use state for alignment assertions."""

    cfg_goids: set[int]
    callgraph_goids: set[int]
    coverage_goids: set[int]
    symbol_use_paths: set[str]


@dataclass
class SpanTestEnv:
    """Reusable environment for span alignment checks."""

    repo_root: Path
    builder: ConfigBuilder
    gateway: StorageGateway
    expected_goid: int | None


@dataclass(frozen=True)
class GraphEngineSeed:
    """Configuration for seeding an NxGraphEngine in tests."""

    repo: str = "test/metrics"
    commit: str = "metrics123"
    repo_root: Path | None = None
    call_graph: nx.DiGraph | None = None
    import_graph: nx.DiGraph | None = None


__all__ = [
    "COMMIT",
    "REPO",
    "GraphEngineSeed",
    "SpanSnapshot",
    "SpanTestEnv",
]
