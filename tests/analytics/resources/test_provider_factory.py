"""Tests for the provider factory.

This module tests:
- ProviderFactory for creating resource providers
- ProviderFactoryOptions configuration
- Registry creation with different provider combinations
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.factory import (
    ProviderFactory,
    ProviderFactoryOptions,
)
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.resources.graphs import GraphProvider
from codeintel.analytics.resources.module_map import ModuleMapProvider
from codeintel.analytics.resources.registry import ResourceRegistry

# Test constants
MAX_FUNCTIONS_TEST_VALUE = 50


@pytest.fixture
def mock_gateway() -> MagicMock:
    """Create a mock StorageGateway.

    Returns
    -------
    MagicMock
        A mock gateway object.
    """
    gateway = MagicMock(spec=["query", "execute", "connection"])
    gateway.connection = MagicMock()
    return gateway


@pytest.fixture
def mock_snapshot() -> MagicMock:
    """Create a mock SnapshotRef.

    Returns
    -------
    MagicMock
        A mock snapshot object.
    """
    snapshot = MagicMock()
    snapshot.version = "v1.0.0"
    snapshot.repo_id = "test-repo"
    return snapshot


def test_provider_factory_options_defaults() -> None:
    """ProviderFactoryOptions has sensible defaults."""
    options = ProviderFactoryOptions()

    assert options.graph_backend is None
    assert options.graph_options is None
    assert options.max_functions is None
    assert options.language is None


def test_provider_factory_options_custom() -> None:
    """ProviderFactoryOptions accepts custom values."""
    mock_backend = MagicMock()
    mock_graph_opts = MagicMock()
    max_funcs = 100
    lang = "python"

    options = ProviderFactoryOptions(
        graph_backend=mock_backend,
        graph_options=mock_graph_opts,
        max_functions=max_funcs,
        language=lang,
    )

    assert options.graph_backend is mock_backend
    assert options.graph_options is mock_graph_opts
    assert options.max_functions == max_funcs
    assert options.language == lang


def test_provider_factory_init(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """ProviderFactory initializes with gateway and snapshot."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    assert factory.gateway is mock_gateway
    assert factory.snapshot is mock_snapshot


def test_provider_factory_with_options(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """ProviderFactory accepts custom options."""
    options = ProviderFactoryOptions(max_functions=MAX_FUNCTIONS_TEST_VALUE, language="python")

    factory = ProviderFactory(mock_gateway, mock_snapshot, options=options)

    assert factory._options.max_functions == MAX_FUNCTIONS_TEST_VALUE  # noqa: SLF001
    assert factory._options.language == "python"  # noqa: SLF001


def test_create_registry_default(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Create registry with default providers (graphs and catalog)."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    registry = factory.create_registry()

    assert isinstance(registry, ResourceRegistry)
    assert registry.has(GraphProvider)
    assert registry.has(CatalogProvider)
    # Default excludes AST, features, module map
    assert not registry.has(AstProvider)
    assert not registry.has(FeaturesProvider)
    assert not registry.has(ModuleMapProvider)


def test_create_registry_no_graphs(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Create registry without graphs."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    registry = factory.create_registry(include_graphs=False)

    assert not registry.has(GraphProvider)
    assert registry.has(CatalogProvider)


def test_create_registry_no_catalog(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Create registry without catalog."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    registry = factory.create_registry(include_catalog=False)

    assert registry.has(GraphProvider)
    assert not registry.has(CatalogProvider)


def test_create_registry_with_asts(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Create registry including AST provider."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    registry = factory.create_registry(include_asts=True)

    assert registry.has(AstProvider)


def test_create_registry_with_features(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Create registry including features provider."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    registry = factory.create_registry(include_features=True)

    assert registry.has(FeaturesProvider)


def test_create_registry_with_module_map(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Create registry including module map provider."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    registry = factory.create_registry(include_module_map=True)

    assert registry.has(ModuleMapProvider)


def test_create_registry_all_providers(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Create registry with all providers enabled."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    registry = factory.create_registry(
        include_graphs=True,
        include_catalog=True,
        include_asts=True,
        include_features=True,
        include_module_map=True,
    )

    assert registry.has(GraphProvider)
    assert registry.has(CatalogProvider)
    assert registry.has(AstProvider)
    assert registry.has(FeaturesProvider)
    assert registry.has(ModuleMapProvider)


def test_make_catalog_provider(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Make catalog provider returns CatalogProvider instance."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_catalog_provider()

    assert isinstance(provider, CatalogProvider)


def test_make_catalog_provider_caches(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Make catalog provider caches the provider instance."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider1 = factory.make_catalog_provider()
    provider2 = factory.make_catalog_provider()

    assert provider1 is provider2


def test_make_catalog_provider_with_catalog(
    mock_gateway: MagicMock, mock_snapshot: MagicMock
) -> None:
    """Make catalog provider with pre-loaded catalog."""
    mock_catalog = MagicMock()
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_catalog_provider(catalog=mock_catalog)

    # Provider wraps the provided catalog
    assert isinstance(provider, CatalogProvider)


def test_make_graph_provider(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Make graph provider returns GraphProvider instance."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_graph_provider()

    assert isinstance(provider, GraphProvider)


def test_make_graph_provider_caches(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Make graph provider caches the provider instance."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider1 = factory.make_graph_provider()
    provider2 = factory.make_graph_provider()

    assert provider1 is provider2


def test_make_graph_provider_with_runtime(
    mock_gateway: MagicMock, mock_snapshot: MagicMock
) -> None:
    """Make graph provider with pre-built runtime."""
    mock_runtime = MagicMock()
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_graph_provider(runtime=mock_runtime)

    # Provider wraps the provided runtime
    assert isinstance(provider, GraphProvider)
    # Does not cache when runtime is provided
    provider2 = factory.make_graph_provider()
    assert provider is not provider2


def test_make_graph_provider_with_options(
    mock_gateway: MagicMock, mock_snapshot: MagicMock
) -> None:
    """Make graph provider with custom options."""
    custom_options = GraphRuntimeOptions(snapshot=mock_snapshot)
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_graph_provider(options=custom_options)

    assert isinstance(provider, GraphProvider)


def test_make_graph_provider_with_backend(
    mock_gateway: MagicMock, mock_snapshot: MagicMock
) -> None:
    """Make graph provider respects factory backend option."""
    mock_backend = MagicMock()
    options = ProviderFactoryOptions(graph_backend=mock_backend)
    factory = ProviderFactory(mock_gateway, mock_snapshot, options=options)

    provider = factory.make_graph_provider()

    assert isinstance(provider, GraphProvider)


def test_make_ast_provider(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Make AST provider returns AstProvider instance."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_ast_provider()

    assert isinstance(provider, AstProvider)


def test_make_ast_provider_with_max_functions(
    mock_gateway: MagicMock, mock_snapshot: MagicMock
) -> None:
    """Make AST provider with custom max_functions."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_ast_provider(max_functions=50)

    assert isinstance(provider, AstProvider)


def test_make_ast_provider_uses_factory_option(
    mock_gateway: MagicMock, mock_snapshot: MagicMock
) -> None:
    """Make AST provider uses factory max_functions option."""
    options = ProviderFactoryOptions(max_functions=100)
    factory = ProviderFactory(mock_gateway, mock_snapshot, options=options)

    provider = factory.make_ast_provider()

    assert isinstance(provider, AstProvider)


def test_make_features_provider(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Make features provider returns FeaturesProvider instance."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_features_provider()

    assert isinstance(provider, FeaturesProvider)


def test_make_features_provider_with_max_functions(
    mock_gateway: MagicMock, mock_snapshot: MagicMock
) -> None:
    """Make features provider with custom max_functions."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_features_provider(max_functions=75)

    assert isinstance(provider, FeaturesProvider)


def test_make_module_map_provider(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Make module map provider returns ModuleMapProvider instance."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_module_map_provider()

    assert isinstance(provider, ModuleMapProvider)


def test_make_module_map_provider_with_language(
    mock_gateway: MagicMock, mock_snapshot: MagicMock
) -> None:
    """Make module map provider with language filter."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    provider = factory.make_module_map_provider(language="python")

    assert isinstance(provider, ModuleMapProvider)


def test_make_module_map_provider_uses_factory_option(
    mock_gateway: MagicMock, mock_snapshot: MagicMock
) -> None:
    """Make module map provider uses factory language option."""
    options = ProviderFactoryOptions(language="typescript")
    factory = ProviderFactory(mock_gateway, mock_snapshot, options=options)

    provider = factory.make_module_map_provider()

    assert isinstance(provider, ModuleMapProvider)


def test_clear_cache(mock_gateway: MagicMock, mock_snapshot: MagicMock) -> None:
    """Clear cache resets cached providers."""
    factory = ProviderFactory(mock_gateway, mock_snapshot)

    # Create cached providers
    catalog1 = factory.make_catalog_provider()
    graphs1 = factory.make_graph_provider()

    # Clear cache
    factory.clear_cache()

    # New providers should be different instances
    catalog2 = factory.make_catalog_provider()
    graphs2 = factory.make_graph_provider()

    assert catalog1 is not catalog2
    assert graphs1 is not graphs2
