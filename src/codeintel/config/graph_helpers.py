"""Graph-related helper types for serving and analytics.

This module contains helper types that were originally in steps_graphs.py
but are still needed by serving backends and analytics code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class GraphMetricWeights:
    """Weights applied to graph metric plugin outputs."""

    pagerank: float = 1.0
    betweenness: float = 1.0
    closeness: float = 1.0
    fan_in: float = 1.0
    fan_out: float = 1.0


@dataclass(frozen=True)
class GraphMetricPluginSelection:
    """Plugin selection policy for graph metrics."""

    enabled: tuple[str, ...] = ()
    disabled: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphMetricPluginOverrides:
    """Overrides for plugin weighting, selection, and options."""

    weights: GraphMetricWeights | None = None
    selection: GraphMetricPluginSelection | None = None
    options: dict[str, dict[str, object]] | None = None


@dataclass(frozen=True)
class GraphPluginPolicy:
    """Execution policy for graph metric plugins."""

    fail_fast: bool = True
    default_severity: Literal["fatal", "soft_fail", "skip_on_error"] = "fatal"
    severity_overrides: dict[str, Literal["fatal", "soft_fail", "skip_on_error"]] = field(
        default_factory=dict
    )
    timeouts_ms: dict[str, int] = field(default_factory=dict)
    skip_on_unchanged: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class GraphRunScope:
    """Optional scoping for incremental graph metric execution."""

    paths: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    time_window: tuple[datetime, datetime] | None = None

    def __post_init__(self) -> None:
        """Normalize iterable inputs to tuples for type stability."""
        if not isinstance(self.paths, tuple):
            object.__setattr__(self, "paths", tuple(self.paths))
        if not isinstance(self.modules, tuple):
            object.__setattr__(self, "modules", tuple(self.modules))


@dataclass(frozen=True)
class GraphMetricsTuning:
    """Tuning parameters for graph metric computation."""

    max_betweenness_sample: int | None = 200
    eigen_max_iter: int = 200
    seed: int = 0


__all__ = [
    "GraphMetricPluginOverrides",
    "GraphMetricPluginSelection",
    "GraphMetricWeights",
    "GraphMetricsTuning",
    "GraphPluginPolicy",
    "GraphRunScope",
]
