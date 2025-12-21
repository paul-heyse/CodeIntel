"""Build-owned analytics resource registry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.factory import ProviderFactory, ProviderFactoryOptions
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.resources.module_map import ModuleMapProvider
from codeintel.build.hamilton.graph_runtime_options import load_graph_runtime_options
from codeintel.core.resources import ResourceRegistry
from codeintel.graphs.resources.graph_provider import GraphProvider

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.graphs.runtime import GraphRuntimeOptions

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyticsResourceOptions:
    """Default options for analytics resource construction."""

    graph_options: GraphRuntimeOptions | None = None
    max_functions: int | None = None
    language: str | None = None


@dataclass(frozen=True)
class AnalyticsResourceIncludes:
    """Toggle which analytics resources to register."""

    include_graphs: bool = True
    include_catalog: bool = True
    include_asts: bool = False
    include_features: bool = False
    include_module_map: bool = False


@dataclass
class AnalyticsResourceRegistryProvider:
    """Factory for ResourceRegistry instances bound to a BuildEnv."""

    default_options: AnalyticsResourceOptions = field(default_factory=AnalyticsResourceOptions)

    def registry_for(
        self,
        env: BuildEnv,
        *,
        target_name: str | None = None,
        include: AnalyticsResourceIncludes | None = None,
        options: AnalyticsResourceOptions | None = None,
    ) -> ResourceRegistry:
        """Create a ResourceRegistry configured for the build environment.

        Parameters
        ----------
        env
            Build environment providing gateway and snapshot info.
        target_name
            Optional target name used to load target-specific graph options.
        include
            Optional include toggles for analytics resources.
        options
            Optional override options for provider construction.

        Returns
        -------
        ResourceRegistry
            Registry populated with the requested providers.
        """
        resolved = options or self.default_options
        includes = include or AnalyticsResourceIncludes()
        graph_options = resolved.graph_options
        if graph_options is None and includes.include_graphs and target_name is not None:
            try:
                graph_options = load_graph_runtime_options(env, target_name=target_name)
            except (RuntimeError, TypeError, ValueError) as exc:
                log.warning(
                    "Failed to load graph runtime options for %s: %s",
                    target_name,
                    exc,
                )
                graph_options = None

        factory_options = ProviderFactoryOptions(
            graph_options=graph_options,
            max_functions=resolved.max_functions,
            language=resolved.language,
        )
        factory = ProviderFactory(env.gateway, env.snapshot, options=factory_options)

        registry = ResourceRegistry()
        catalog_provider: CatalogProvider | None = None
        catalog: FunctionCatalogProvider | None = None

        if includes.include_catalog:
            catalog_provider = factory.make_catalog_provider()
            registry.register(CatalogProvider, catalog_provider)
            catalog = catalog_provider.get()

        if includes.include_graphs:
            registry.register(GraphProvider, factory.make_graph_provider())

        if includes.include_asts:
            registry.register(
                AstProvider,
                factory.make_ast_provider(catalog_provider=catalog),
            )

        if includes.include_features:
            registry.register(
                FeaturesProvider,
                factory.make_features_provider(catalog_provider=catalog),
            )

        if includes.include_module_map:
            registry.register(ModuleMapProvider, factory.make_module_map_provider())

        return registry


__all__ = [
    "AnalyticsResourceIncludes",
    "AnalyticsResourceOptions",
    "AnalyticsResourceRegistryProvider",
]
