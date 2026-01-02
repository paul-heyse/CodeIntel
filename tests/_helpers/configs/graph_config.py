"""Configuration dataclasses for graph test environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.fixtures.snapshots import METRICS_VARIANT, SnapshotVariant

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class SpanSnapshot:
    """Collected GOID/symbol-use state for alignment assertions."""

    cfg_goids: set[int]
    callgraph_goids: set[int]
    symbol_use_paths: set[str]


@dataclass
class SpanTestEnv:
    """Reusable environment for span alignment checks."""

    repo_root: Path
    snapshot: SnapshotRef
    gateway: StorageGateway
    expected_goid: int | None


@dataclass(frozen=True)
class GraphEngineSeed:
    """Configuration for seeding an NxGraphEngine in tests."""

    snapshot_variant: SnapshotVariant = METRICS_VARIANT
    repo_root: Path | None = None
    call_graph: nx.DiGraph | None = None
    import_graph: nx.DiGraph | None = None

    @property
    def repo(self) -> str:
        return self.snapshot_variant.repo

    @property
    def commit(self) -> str:
        return self.snapshot_variant.commit


__all__ = [
    "GraphEngineSeed",
    "SpanSnapshot",
    "SpanTestEnv",
]
