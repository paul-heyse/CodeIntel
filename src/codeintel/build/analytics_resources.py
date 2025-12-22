"""Build-owned analytics resource registry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.factory import (
    ProviderFactory,
    ProviderFactoryOptions,
    ProviderRegistryOptions,
)
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.resources.module_map import ModuleMapProvider
from codeintel.build.hamilton.graph_runtime_options import load_graph_runtime_options
from codeintel.core.resources import ResourceRegistry
from codeintel.graphs.resources.graph_provider import GraphProvider

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv

log = logging.getLogger(__name__)


@dataclass
class AnalyticsResourceRegistryProvider:
    """Factory for ResourceRegistry instances bound to a BuildEnv."""

    default_factory_options: ProviderFactoryOptions = field(default_factory=ProviderFactoryOptions)

    def registry_for(
        self,
        env: BuildEnv,
        *,
        target_name: str | None = None,
        options: ProviderRegistryOptions | None = None,
        factory_options: ProviderFactoryOptions | None = None,
    ) -> ResourceRegistry:
        """Create a ResourceRegistry configured for the build environment.

        Parameters
        ----------
        env
            Build environment providing gateway and snapshot info.
        target_name
            Optional target name used to load target-specific graph options.
        options
            Optional provider registry selection options.
        factory_options
            Optional factory override options for provider construction.

        Returns
        -------
        ResourceRegistry
            Registry populated with the requested providers.
        """
        resolved_registry = options or ProviderRegistryOptions()
        resolved_factory = factory_options or self.default_factory_options
        graph_options = resolved_factory.graph_options
        if graph_options is None and resolved_registry.include_graphs and target_name is not None:
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
            graph_backend=resolved_factory.graph_backend,
            max_functions=resolved_factory.max_functions,
            language=resolved_factory.language,
        )
        factory = ProviderFactory(env.gateway, env.snapshot, options=factory_options)

        registry = ResourceRegistry()
        catalog_provider: CatalogProvider | None = None
        catalog = None

        if resolved_registry.include_catalog:
            catalog_provider = factory.make_catalog_provider()
            registry.register(CatalogProvider, catalog_provider)
            catalog = catalog_provider.get()

        if resolved_registry.include_graphs:
            registry.register(GraphProvider, factory.make_graph_provider(options=graph_options))

        if resolved_registry.include_asts:
            registry.register(
                AstProvider,
                factory.make_ast_provider(
                    catalog_provider=catalog,
                    max_functions=resolved_factory.max_functions,
                ),
            )

        if resolved_registry.include_features:
            registry.register(
                FeaturesProvider,
                factory.make_features_provider(
                    catalog_provider=catalog,
                    max_functions=resolved_factory.max_functions,
                ),
            )

        if resolved_registry.include_module_map:
            registry.register(
                ModuleMapProvider,
                factory.make_module_map_provider(language=resolved_factory.language),
            )

        return registry


__all__ = [
    "AnalyticsResourceRegistryProvider",
]
