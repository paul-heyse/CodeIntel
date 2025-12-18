"""Unified runtime primitive bundle.

This module defines a single, typed bundle of runtime primitives that can be
constructed by both CLI resolution and config builders.
"""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.config.primitives import (
    BuildPaths,
    GraphBackendConfig,
    GraphFeatureFlags,
    ScanProfiles,
    SnapshotRef,
)
from codeintel.core.tools import ToolBinaries


@dataclass(frozen=True)
class RuntimePrimitives:
    """Canonical bundle of runtime primitives used across the repo.

    Parameters
    ----------
    snapshot
        Repository snapshot identity for the run.
    paths
        Derived build paths for the run.
    tools
        Tool executable configuration.
    graph_backend
        Graph backend selection configuration.
    graph_features
        Graph runtime feature flags.
    profiles
        Optional scan profile bundle when a run is using scan-based ingestion.
    """

    snapshot: SnapshotRef
    paths: BuildPaths
    tools: ToolBinaries
    graph_backend: GraphBackendConfig
    graph_features: GraphFeatureFlags
    profiles: ScanProfiles | None = None


__all__ = ["RuntimePrimitives"]
