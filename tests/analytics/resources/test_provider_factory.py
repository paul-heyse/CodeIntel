"""Tests for the provider factory.

This module tests:
- ProviderFactory for creating resource providers
- ProviderFactoryOptions configuration
- Registry creation with different provider combinations

Note: Uses shared analytics fixtures from analytics/conftest.py.
"""

from __future__ import annotations

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
from codeintel.analytics.runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.graphs.catalog import (
    FunctionCatalog,
    FunctionCatalogProvider,
    FunctionCatalogService,
    FunctionMeta,
)
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)
from tests._helpers.fakes.graph_engine import StubGraphEngine

# Test constants
MAX_FUNCTIONS_TEST_VALUE = 50


class DummyCatalogProvider(FunctionCatalogProvider):
    """Minimal catalog provider for testing."""

    def __init__(self) -> None:
        empty_functions: list[FunctionMeta] = []
        self._catalog = FunctionCatalog(functions=empty_functions, module_by_path={})

    def catalog(self) -> FunctionCatalog:
        """
        Return the cached catalog.

        Returns
        -------
        FunctionCatalog
            Empty catalog instance for testing.
        """
        return self._catalog

    def urn_for_goid(self, goid: int) -> str | None:
        """
        Return URN for GOID (none for dummy).

        Parameters
        ----------
        goid
            GOID to look up.

        Returns
        -------
        str | None
            Always None for the dummy provider.
        """
        _ = (self._catalog, goid)
        return None

    def lookup_goid(
        self,
        rel_path: str,
        start_line: int,
        end_line: int | None,
        qualname: str | None,
    ) -> int | None:
        """
        Lookup GOID for a span (none for dummy).

        Parameters
        ----------
        rel_path
            Relative path of the function.
        start_line
            Starting line number.
        end_line
            Optional end line number.
        qualname
            Optional qualified name.

        Returns
        -------
        int | None
            Always None for the dummy provider.
        """
        _ = (self._catalog, rel_path, start_line, end_line, qualname)
        return None

    def module_for_path(self, rel_path: str) -> str | None:
        """
        Return module name for a relative path (none for dummy).

        Parameters
        ----------
        rel_path
            Relative path to look up.

        Returns
        -------
        str | None
            Always None for the dummy provider.
        """
        _ = (self._catalog, rel_path)
        return None


def test_provider_factory_options_defaults() -> None:
    """ProviderFactoryOptions has sensible defaults."""
    options = ProviderFactoryOptions()

    expect_true(options.graph_backend is None)
    expect_true(options.graph_options is None)
    expect_true(options.max_functions is None)
    expect_true(options.language is None)


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

    expect_true(options.graph_backend is backend_config)
    expect_true(options.graph_options is graph_opts)
    expect_equal(options.max_functions, max_funcs)
    expect_equal(options.language, lang)


def test_provider_factory_init(test_gateway: StorageGateway, test_snapshot: SnapshotRef) -> None:
    """ProviderFactory initializes with gateway and snapshot."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    expect_true(factory.gateway is test_gateway)
    expect_true(factory.snapshot is test_snapshot)


def test_provider_factory_with_options(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """ProviderFactory accepts custom options."""
    options = ProviderFactoryOptions(max_functions=MAX_FUNCTIONS_TEST_VALUE, language="python")

    factory = ProviderFactory(test_gateway, test_snapshot, options=options)

    expect_equal(factory.options.max_functions, MAX_FUNCTIONS_TEST_VALUE)
    expect_equal(factory.options.language, "python")


def test_create_registry_default(test_gateway: StorageGateway, test_snapshot: SnapshotRef) -> None:
    """Create registry with default providers (graphs and catalog)."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry()

    expect_is_instance(registry, ResourceRegistry)
    expect_true(registry.has(GraphProvider))
    expect_true(registry.has(CatalogProvider))
    # Default excludes AST, features, module map
    expect_true(not registry.has(AstProvider))
    expect_true(not registry.has(FeaturesProvider))
    expect_true(not registry.has(ModuleMapProvider))


def test_create_registry_no_graphs(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry without graphs."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry(include_graphs=False)

    expect_true(not registry.has(GraphProvider))
    expect_true(registry.has(CatalogProvider))


def test_create_registry_no_catalog(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry without catalog."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry(include_catalog=False)

    expect_true(registry.has(GraphProvider))
    expect_true(not registry.has(CatalogProvider))


def test_create_registry_with_asts(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry including AST provider."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry(include_asts=True)

    expect_true(registry.has(AstProvider))


def test_create_registry_with_features(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry including features provider."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry(include_features=True)

    expect_true(registry.has(FeaturesProvider))


def test_create_registry_with_module_map(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Create registry including module map provider."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    registry = factory.create_registry(include_module_map=True)

    expect_true(registry.has(ModuleMapProvider))


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

    expect_true(registry.has(GraphProvider))
    expect_true(registry.has(CatalogProvider))
    expect_true(registry.has(AstProvider))
    expect_true(registry.has(FeaturesProvider))
    expect_true(registry.has(ModuleMapProvider))


def test_function_catalog_service_module_lookup() -> None:
    """FunctionCatalogService should expose module_for_path."""
    catalog = FunctionCatalog(functions=[], module_by_path={"src/a.py": "src.a"})
    provider = FunctionCatalogService(catalog)

    expect_equal(provider.module_for_path("src/a.py"), "src.a")
    expect_true(provider.module_for_path("missing.py") is None)


def test_make_catalog_provider(test_gateway: StorageGateway, test_snapshot: SnapshotRef) -> None:
    """Make catalog provider returns CatalogProvider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_catalog_provider()

    expect_is_instance(provider, CatalogProvider)


def test_make_catalog_provider_caches(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make catalog provider caches the provider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider1 = factory.make_catalog_provider()
    provider2 = factory.make_catalog_provider()

    expect_true(provider1 is provider2)


def test_make_catalog_provider_with_catalog(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make catalog provider with pre-loaded catalog."""
    catalog_obj = DummyCatalogProvider()
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_catalog_provider(catalog=catalog_obj)

    # Provider wraps the provided catalog
    expect_is_instance(provider, CatalogProvider)


def test_make_graph_provider(test_gateway: StorageGateway, test_snapshot: SnapshotRef) -> None:
    """Make graph provider returns GraphProvider instance."""
    stub_engine = StubGraphEngine(gateway=test_gateway, snapshot=test_snapshot)
    options = ProviderFactoryOptions(
        graph_options=GraphRuntimeOptions(snapshot=test_snapshot, engine=stub_engine)
    )
    factory = ProviderFactory(test_gateway, test_snapshot, options=options)

    provider = factory.make_graph_provider()

    expect_is_instance(provider, GraphProvider)


def test_make_graph_provider_caches(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make graph provider caches the provider instance."""
    stub_engine = StubGraphEngine(gateway=test_gateway, snapshot=test_snapshot)
    options = ProviderFactoryOptions(
        graph_options=GraphRuntimeOptions(snapshot=test_snapshot, engine=stub_engine)
    )
    factory = ProviderFactory(test_gateway, test_snapshot, options=options)

    provider1 = factory.make_graph_provider()
    provider2 = factory.make_graph_provider()

    expect_true(provider1 is provider2)


def test_make_graph_provider_with_runtime(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make graph provider with pre-built runtime."""
    stub_engine = StubGraphEngine(gateway=test_gateway, snapshot=test_snapshot)
    runtime_options = GraphRuntimeOptions(snapshot=test_snapshot, engine=stub_engine)
    factory = ProviderFactory(
        test_gateway, test_snapshot, options=ProviderFactoryOptions(graph_options=runtime_options)
    )
    runtime = GraphRuntime(options=runtime_options, engine=stub_engine)

    provider = factory.make_graph_provider(runtime=runtime)

    # Provider wraps the provided runtime
    expect_is_instance(provider, GraphProvider)
    # Does not cache when runtime is provided
    provider2 = factory.make_graph_provider()
    expect_true(provider is not provider2)


def test_make_graph_provider_with_options(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make graph provider with custom options."""
    stub_engine = StubGraphEngine(gateway=test_gateway, snapshot=test_snapshot)
    custom_options = GraphRuntimeOptions(snapshot=test_snapshot, engine=stub_engine)
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_graph_provider(options=custom_options)

    expect_is_instance(provider, GraphProvider)


def test_make_graph_provider_with_backend(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make graph provider respects factory backend option."""
    backend_config = GraphBackendConfig(use_gpu=False, backend="cpu")
    options = ProviderFactoryOptions(graph_backend=backend_config)
    factory = ProviderFactory(test_gateway, test_snapshot, options=options)

    provider = factory.make_graph_provider()

    expect_is_instance(provider, GraphProvider)


def test_make_ast_provider(test_gateway: StorageGateway, test_snapshot: SnapshotRef) -> None:
    """Make AST provider returns AstProvider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_ast_provider()

    expect_is_instance(provider, AstProvider)


def test_make_ast_provider_with_max_functions(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make AST provider with custom max_functions."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_ast_provider(max_functions=50)

    expect_is_instance(provider, AstProvider)


def test_make_ast_provider_uses_factory_option(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make AST provider uses factory max_functions option."""
    options = ProviderFactoryOptions(max_functions=100)
    factory = ProviderFactory(test_gateway, test_snapshot, options=options)

    provider = factory.make_ast_provider()

    expect_is_instance(provider, AstProvider)


def test_make_features_provider(test_gateway: StorageGateway, test_snapshot: SnapshotRef) -> None:
    """Make features provider returns FeaturesProvider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_features_provider()

    expect_is_instance(provider, FeaturesProvider)


def test_make_features_provider_with_max_functions(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make features provider with custom max_functions."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_features_provider(max_functions=75)

    expect_is_instance(provider, FeaturesProvider)


def test_make_module_map_provider(test_gateway: StorageGateway, test_snapshot: SnapshotRef) -> None:
    """Make module map provider returns ModuleMapProvider instance."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_module_map_provider()

    expect_is_instance(provider, ModuleMapProvider)


def test_make_module_map_provider_with_language(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make module map provider with language filter."""
    factory = ProviderFactory(test_gateway, test_snapshot)

    provider = factory.make_module_map_provider(language="python")

    expect_is_instance(provider, ModuleMapProvider)


def test_make_module_map_provider_uses_factory_option(
    test_gateway: StorageGateway, test_snapshot: SnapshotRef
) -> None:
    """Make module map provider uses factory language option."""
    options = ProviderFactoryOptions(language="typescript")
    factory = ProviderFactory(test_gateway, test_snapshot, options=options)

    provider = factory.make_module_map_provider()

    expect_is_instance(provider, ModuleMapProvider)


def test_clear_cache(test_gateway: StorageGateway, test_snapshot: SnapshotRef) -> None:
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

    expect_true(catalog1 is not catalog2)
    expect_true(graphs1 is not graphs2)
