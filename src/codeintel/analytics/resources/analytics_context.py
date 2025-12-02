"""Analytics context resource provider.

This module provides `AnalyticsContextProvider` which lazily builds
and caches the legacy `AnalyticsContext` for plugins that still need it.

The provider wraps the existing `build_analytics_context` function and
integrates it with the ResourceRegistry pattern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.analytics.resources.protocol import LazyResource

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext, AnalyticsContextConfig
    from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions
    from codeintel.graphs.engine import GraphEngine
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class AnalyticsContextProvider(LazyResource["AnalyticsContext"]):
    """Provider that lazily builds AnalyticsContext.

    This provider wraps the legacy `build_analytics_context` function
    and integrates it with the ResourceRegistry pattern. Use this to
    provide AnalyticsContext to plugins through the resource system.

    Example
    -------
    >>> provider = AnalyticsContextProvider(gateway, config)
    >>> registry.register(AnalyticsContextProvider, provider)
    >>> ctx_plugin = registry.require(AnalyticsContextProvider)
    """

    def __init__(
        self,
        gateway: StorageGateway,
        config: AnalyticsContextConfig,
        *,
        runtime: GraphRuntime | GraphRuntimeOptions | None = None,
        engine: GraphEngine | None = None,
    ) -> None:
        """Initialize the provider.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        config
            Analytics context configuration.
        runtime
            Optional graph runtime or options.
        engine
            Optional pre-built graph engine.
        """
        super().__init__("AnalyticsContext")
        self._gateway = gateway
        self._config = config
        self._runtime = runtime
        self._engine = engine

    def _load(self) -> AnalyticsContext:
        """Load the AnalyticsContext.

        Returns
        -------
        AnalyticsContext
            The built analytics context.
        """
        from codeintel.analytics.context import build_analytics_context

        log.debug(
            "Building AnalyticsContext for %s@%s",
            self._config.repo,
            self._config.commit,
        )

        return build_analytics_context(
            self._gateway,
            self._config,
            runtime=self._runtime,
            engine=self._engine,
        )

    @property
    def config(self) -> AnalyticsContextConfig:
        """Return the context configuration.

        Returns
        -------
        AnalyticsContextConfig
            The configuration used to build the context.
        """
        return self._config


__all__ = ["AnalyticsContextProvider"]

