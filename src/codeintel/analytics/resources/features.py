"""Function AST features resource provider.

This module provides `FeaturesProvider` for lazy loading of function
AST features used in analytics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.analytics.ast_features.extract import compute_function_features
from codeintel.analytics.function_ast_cache import (
    FunctionAstLoadRequest,
    load_function_asts,
)
from codeintel.analytics.resources.protocol import LazyResource, ResourceNotLoadedError

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.config.primitives import SnapshotRef
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class FeaturesProvider(LazyResource[dict[int, "FunctionAstFeatures"]]):
    """Provider for function AST features with lazy loading.

    Computes feature vectors (IO flags, decorators, library usage, etc.)
    for all functions in a repository snapshot.

    Parameters can be None when using factory methods like `from_features()`
    that set a pre-loaded resource.

    Example
    -------
    >>> provider = FeaturesProvider(gateway, snapshot)
    >>> features = provider.get()
    >>> func_features = features.get(function_goid)
    """

    def __init__(
        self,
        gateway: StorageGateway | None = None,
        snapshot: SnapshotRef | None = None,
        *,
        catalog_provider: FunctionCatalogProvider | None = None,
        max_functions: int | None = None,
    ) -> None:
        """Initialize the features provider.

        Parameters
        ----------
        gateway
            Storage gateway for database access. Can be None if using
            `set_preloaded()` or `from_features()` factory method.
        snapshot
            Repository snapshot reference. Can be None if using
            `set_preloaded()` or `from_features()` factory method.
        catalog_provider
            Optional pre-loaded catalog provider.
        max_functions
            Maximum number of functions to process.
        """
        super().__init__("FunctionFeatures")
        self._gateway = gateway
        self._snapshot = snapshot
        self._catalog_provider = catalog_provider
        self._max_functions = max_functions

    @classmethod
    def from_features(
        cls,
        features: dict[int, FunctionAstFeatures],
    ) -> FeaturesProvider:
        """Create a provider from existing features.

        Use this factory when features have already been computed and you
        want to wrap them in a provider for the resource registry.

        Parameters
        ----------
        features
            Pre-computed function features map.

        Returns
        -------
        FeaturesProvider
            Provider wrapping the existing features.

        Example
        -------
        >>> existing_features = context.function_features_map
        >>> provider = FeaturesProvider.from_features(existing_features)
        >>> registry.register(FeaturesProvider, provider)
        """
        # Create provider with None - valid since we set preloaded
        provider = cls(gateway=None, snapshot=None)
        provider.set_preloaded(features)
        return provider

    def _load(self) -> dict[int, FunctionAstFeatures]:
        """Load and compute function features.

        Returns
        -------
        dict[int, FunctionAstFeatures]
            Mapping of GOID to feature vector.

        Raises
        ------
        ResourceNotLoadedError
            If gateway or snapshot are None (provider created for pre-loading only).
        """
        if self._gateway is None or self._snapshot is None:
            raise ResourceNotLoadedError(
                self._name,
                "Cannot load - provider was created for pre-loaded resource only. "
                "Use from_features() with pre-loaded data or provide gateway and snapshot.",
            )

        request = FunctionAstLoadRequest(
            repo=self._snapshot.repo,
            commit=self._snapshot.commit,
            repo_root=self._snapshot.repo_root,
            catalog_provider=self._catalog_provider,
            max_functions=self._max_functions,
        )

        function_ast_map, missing = load_function_asts(self._gateway, request)

        if missing:
            log.debug(
                "Skipped %d functions without AST spans during feature extraction",
                len(missing),
            )

        features: dict[int, FunctionAstFeatures] = {}
        repo_root = self._snapshot.repo_root

        for goid, fn_ast in function_ast_map.items():
            features[goid] = compute_function_features(fn_ast, repo_root=repo_root)

        log.debug(
            "Computed features for %d functions in %s@%s",
            len(features),
            self._snapshot.repo,
            self._snapshot.commit,
        )

        return features

    @property
    def function_features(self) -> dict[int, FunctionAstFeatures]:
        """Return function features map.

        Convenience property matching the legacy AnalyticsContext API.

        Returns
        -------
        dict[int, FunctionAstFeatures]
            Mapping of GOID to feature vector.
        """
        return self.get()


__all__ = [
    "FeaturesProvider",
]
