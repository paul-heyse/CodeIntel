"""Typed builders for snapshot and graph runtime options used in tests."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Unpack

from codeintel.analytics.runtime import GraphRuntimeOptions
from codeintel.config import GraphBackendConfig
from codeintel.config.primitives import GraphFeatureFlags, SnapshotRef
from codeintel.graphs.engine import GraphEngine, GraphKind


def make_snapshot(
    repo: str = "demo/repo",
    commit: str = "deadbeef",
    repo_root: Path | None = None,
) -> SnapshotRef:
    """
    Build a SnapshotRef with sensible defaults.

    Parameters
    ----------
    repo
        Repository slug.
    commit
        Commit SHA or identifier.
    repo_root
        Optional root path override; defaults to current working directory.

    Returns
    -------
    SnapshotRef
        Snapshot with provided identifiers and root path.
    """
    root = repo_root if repo_root is not None else Path.cwd()
    return SnapshotRef(repo=repo, commit=commit, repo_root=root)


class GraphRuntimeOptionsKwargs(TypedDict, total=False):
    """Typed overrides for GraphRuntimeOptions construction."""

    backend: GraphBackendConfig | None
    graphs: GraphKind
    eager: bool
    validate: bool
    cache_key: str | None
    engine: GraphEngine | None
    graph_cache_dir: Path | None
    features: GraphFeatureFlags


def make_graph_runtime_options(
    snapshot: SnapshotRef | None = None,
    **overrides: Unpack[GraphRuntimeOptionsKwargs],
) -> GraphRuntimeOptions:
    """
    Build GraphRuntimeOptions with a provided or default snapshot.

    Parameters
    ----------
    snapshot
        Snapshot to attach; defaults to a demo snapshot when omitted.
    overrides
        Typed keyword overrides forwarded to GraphRuntimeOptions.

    Returns
    -------
    GraphRuntimeOptions
        Options instance ready for tests.
    """
    effective_snapshot = snapshot or make_snapshot()
    if "features" not in overrides:
        overrides["features"] = GraphFeatureFlags()
    return GraphRuntimeOptions(snapshot=effective_snapshot, **overrides)


__all__ = [
    "GraphRuntimeOptionsKwargs",
    "make_graph_runtime_options",
    "make_snapshot",
]
