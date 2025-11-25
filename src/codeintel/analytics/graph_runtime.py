"""Shared graph runtime options for analytics modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.graph_service import GraphContext
from codeintel.config.models import GraphBackendConfig

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext


@dataclass(frozen=True)
class GraphRuntimeOptions:
    """
    Runtime options for graph-based analytics functions.

    Attributes
    ----------
    context :
        Optional shared AnalyticsContext containing cached graphs.
    graph_ctx :
        Optional GraphContext carrying weights and sampling limits.
    graph_backend :
        Optional backend selection used to derive GPU preferences.
    """

    context: AnalyticsContext | None = None
    graph_ctx: GraphContext | None = None
    graph_backend: GraphBackendConfig | None = None

    @property
    def use_gpu(self) -> bool:
        """
        Compute whether GPU execution is preferred.

        Returns
        -------
        bool
            True when graph_backend or graph_ctx requests GPU usage.
        """
        if self.graph_backend is not None:
            return bool(self.graph_backend.use_gpu)
        if self.graph_ctx is not None:
            return bool(self.graph_ctx.use_gpu)
        return False
