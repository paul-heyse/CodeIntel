"""Tests for the provider factory.

This module tests:
- ProviderFactory for creating resource providers
- ProviderFactoryOptions configuration
- Registry creation with different provider combinations
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

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
from codeintel.analytics.runtime import GraphRuntimeOptions
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from tests._helpers.fakes.configs import create_test_snapshot

# Test constants
MAX_FUNCTIONS_TEST_VALUE = 50


@pytest.fixture
def test_gateway() -> Generator[StorageGateway]:
    """Create an in-memory StorageGateway for testing.

    Yields
    ------
    StorageGateway
        An in-memory gateway with schema applied.
    """
    gateway = open_memory_gateway(validate_schema=False)
    yield gateway
    gateway.close()


@pytest.fixture
def test_snapshot(tmp_path: Path) -> SnapshotRef:
    """Create a real SnapshotRef for testing.

    Parameters
    ----------
    tmp_path
        Pytest temporary path fixture.

    Returns
    -------
    SnapshotRef
        A configured snapshot reference.
    """
    return create_test_snapshot(tmp_path)


def test_provider_factory_options_defaults() -> None:
    """ProviderFactoryOptions has sensible defaults."""
    options = ProviderFactoryOptions()

    assert options.graph_backend is None
    assert options.graph_options is None
    assert options.max_functions is None
    assert options.language is None


def test_provider_factory_options_custom() -> None:
    """ProviderFactoryOptions accepts custom values."""
    backend_config = GraphBackendConfig(use_gpu=False, backend="cpu")
    graph_opts = GraphRuntimeOptions(eager=True, validate=True)
    max_funcs = 100
    lang = "python"

    options = ProviderFactoryOptions(
        graph_backend=backend_config,
        graph_options=graph_opts,
        max_functions=max_funcs,
        language=lang,
    )

    assert options.graph_backend is backend_config
    assert options.graph_options is graph_opts
    assert options.max_functions == max_funcs
    assert options.language == lang


def test_provider_factory_init(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """ProviderFactory initializes with gateway and snapshot."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    assert factory.gateway is test_gateway
    assert factory.snapshot is test_snapshot


def test_provider_factory_with_options(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """ProviderFactory accepts custom options."""
    options = ProviderFactoryOptions(max_functions=MAX_FUNCTIONS_TEST_VALUE, language="python")

    factory = ProviderFactory(test_gateway, test_snapshot, options=options)

    assert factory._options.max_functions == MAX_FUNCTIONS_TEST_VALUE  # noqa: SLF001
    assert factory._options.language == "python"  # noqa: SLF001


def test_create_registry_default(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry with default providers (graphs and catalog)."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry()

    assert isinstance(registry, ResourceRegistry)
    assert registry.has(GraphProvider)
    assert registry.has(CatalogProvider)
    # Default excludes AST, features, module map
    assert not registry.has(AstProvider)
    assert not registry.has(FeaturesProvider)
    assert not registry.has(ModuleMapProvider)


def test_create_registry_no_graphs(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry without graphs."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry(include_graphs=False)

    assert not registry.has(GraphProvider)
    assert registry.has(CatalogProvider)


def test_create_registry_no_catalog(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry without catalog."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry(include_catalog=False)

    assert registry.has(GraphProvider)
    assert not registry.has(CatalogProvider)


def test_create_registry_with_asts(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry including AST provider."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry(include_asts=True)

    assert registry.has(AstProvider)


def test_create_registry_with_features(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry including features provider."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry(include_features=True)

    assert registry.has(FeaturesProvider)


def test_create_registry_with_module_map(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry including module map provider."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry(include_module_map=True)

    assert registry.has(ModuleMapProvider)


def test_create_registry_all_providers(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry with all providers enabled."""
    factory = ProviderFactory(test_gateway, test_snapshot)

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


def test_make_catalog_provider(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make catalog provider returns CatalogProvider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_catalog_provider()

    assert isinstance(provider, CatalogProvider)


def test_make_catalog_provider_caches(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make catalog provider caches the provider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider1 = factory.make_catalog_provider()
    provider2 = factory.make_catalog_provider()

    assert provider1 is provider2


def test_make_catalog_provider_with_catalog(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make catalog provider with pre-loaded catalog."""
    catalog_obj = object()  # Use plain object as placeholder
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_catalog_provider(catalog=catalog_obj)  # type: ignore[arg-type]

    # Provider wraps the provided catalog
    assert isinstance(provider, CatalogProvider)


def test_make_graph_provider(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make graph provider returns GraphProvider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_graph_provider()

    assert isinstance(provider, GraphProvider)


def test_make_graph_provider_caches(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make graph provider caches the provider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider1 = factory.make_graph_provider()
    provider2 = factory.make_graph_provider()

    assert provider1 is provider2


def test_make_graph_provider_with_runtime(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make graph provider with pre-built runtime."""
    runtime_obj = object()  # Use plain object as placeholder
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_graph_provider(runtime=runtime_obj)  # type: ignore[arg-type]

    # Provider wraps the provided runtime
    assert isinstance(provider, GraphProvider)
    # Does not cache when runtime is provided
    provider2 = factory.make_graph_provider()
    assert provider is not provider2


def test_make_graph_provider_with_options(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make graph provider with custom options."""
    custom_options = GraphRuntimeOptions(snapshot=test_snapshot)
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_graph_provider(options=custom_options)

    assert isinstance(provider, GraphProvider)


def test_make_graph_provider_with_backend(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make graph provider respects factory backend option."""
    backend_config = GraphBackendConfig(use_gpu=False, backend="cpu")
    options = ProviderFactoryOptions(graph_backend=backend_config)
    factory = ProviderFactory(test_gateway, test_snapshot, options=options)

    provider = factory.make_graph_provider()

    assert isinstance(provider, GraphProvider)


def test_make_ast_provider(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make AST provider returns AstProvider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_ast_provider()

    assert isinstance(provider, AstProvider)


def test_make_ast_provider_with_max_functions(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make AST provider with custom max_functions."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_ast_provider(max_functions=50)

    assert isinstance(provider, AstProvider)


def test_make_ast_provider_uses_factory_option(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make AST provider uses factory max_functions option."""
    options = ProviderFactoryOptions(max_functions=100)
    factory = ProviderFactory(test_gateway, test_snapshot, options=options)

    provider = factory.make_ast_provider()

    assert isinstance(provider, AstProvider)


def test_make_features_provider(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make features provider returns FeaturesProvider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_features_provider()

    assert isinstance(provider, FeaturesProvider)


def test_make_features_provider_with_max_functions(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make features provider with custom max_functions."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_features_provider(max_functions=75)

    assert isinstance(provider, FeaturesProvider)


def test_make_module_map_provider(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make module map provider returns ModuleMapProvider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_module_map_provider()

    assert isinstance(provider, ModuleMapProvider)


def test_make_module_map_provider_with_language(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make module map provider with language filter."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_module_map_provider(language="python")

    assert isinstance(provider, ModuleMapProvider)


def test_make_module_map_provider_uses_factory_option(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make module map provider uses factory language option."""
    options = ProviderFactoryOptions(language="typescript")
    factory = ProviderFactory(test_gateway, test_snapshot, options=options)

    provider = factory.make_module_map_provider()

    assert isinstance(provider, ModuleMapProvider)


def test_clear_cache(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Clear cache resets cached providers."""
    factory = ProviderFactory(test_gateway, test_snapshot)

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
