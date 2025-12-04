"""Tests for ingestion resource registry and providers.

This module tests the resource registry and provider infrastructure
used to manage dependencies during ingestion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

import pytest

from codeintel.config import SnapshotRef
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.infrastructure.scanning import ScanProfile
from codeintel.ingestion.resources.protocol import (
    LazyResource as LazyResourceBase,
)
from codeintel.ingestion.resources.protocol import (
    ResourceError,
    ResourceNotLoadedError,
    ResourceProvider,
)
from codeintel.ingestion.resources.registry import (
    ResourceNotFoundError,
    ResourceRegistry,
)
from codeintel.ingestion.resources.tools import ToolsProvider
from codeintel.ingestion.resources.tracker import TrackerConfig, TrackerProvider
from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_COUNT_42 = 42
TEST_COUNT_100 = 100
EXPECTED_LEN_3 = 3


# =============================================================================
# Test Providers
# =============================================================================


@dataclass
class TestResource:
    """A simple test resource."""

    value: str
    count: int


class TestProvider(ResourceProvider[TestResource]):
    """Provider that returns a TestResource."""

    RESOURCE_NAME: ClassVar[str] = "test_resource"

    def __init__(self, value: str = "test", count: int = 10) -> None:
        """Initialize with test data."""
        self._value = value
        self._count = count
        self._loaded = False

    def get(self) -> TestResource:
        """Return the test resource.

        Returns
        -------
        TestResource
            The test resource instance.
        """
        self._loaded = True
        return TestResource(value=self._value, count=self._count)

    @property
    def is_loaded(self) -> bool:
        """Return whether the resource has been loaded."""
        return self._loaded

    def get_or_none(self) -> TestResource | None:
        """Return the resource without marking it loaded.

        Returns
        -------
        TestResource | None
            The test resource or None.
        """
        return TestResource(value=self._value, count=self._count)

    def invalidate(self) -> None:
        """Reset the loaded flag."""
        self._loaded = False  # Modifies instance state


class ListProvider(ResourceProvider[Sequence[str]]):
    """Provider that returns a list of strings."""

    RESOURCE_NAME: ClassVar[str] = "list_resource"

    def __init__(self, items: Sequence[str] | None = None) -> None:
        """Initialize with optional items."""
        self._items = list(items or [])
        self._loaded = False

    def get(self) -> Sequence[str]:
        """Return the list of items.

        Returns
        -------
        Sequence[str]
            The list of items.
        """
        self._loaded = True
        return self._items

    @property
    def is_loaded(self) -> bool:
        """Return whether the items have been fetched."""
        return self._loaded

    def get_or_none(self) -> Sequence[str] | None:
        """Return items without forcing load semantics.

        Returns
        -------
        Sequence[str] | None
            The items or None.
        """
        return self._items

    def invalidate(self) -> None:
        """Reset the loaded flag."""
        self._loaded = False  # Modifies instance state


class FailingProvider(ResourceProvider[str]):
    """Provider that always fails."""

    RESOURCE_NAME: ClassVar[str] = "failing_resource"

    def get(self) -> str:
        """Raise an error on access.

        Raises
        ------
        ResourceError
            Always raised.
        """
        _ = self  # Use self for PLR6301
        msg = "Resource unavailable"
        raise ResourceError(msg)

    @property
    def is_loaded(self) -> bool:
        """Return whether the resource has been loaded."""
        return False

    def get_or_none(self) -> str | None:
        """Return None because the resource always fails.

        Returns
        -------
        str | None
            Always returns None.
        """
        _ = self  # Use self for PLR6301
        return None

    def invalidate(self) -> None:
        """Failing provider has no cache to clear."""
        _ = self  # Use self for PLR6301


# =============================================================================
# ResourceNotFoundError Tests
# =============================================================================


def test_resource_not_found_error_from_type() -> None:
    """ResourceNotFoundError should accept a type."""
    error = ResourceNotFoundError(TestProvider)

    assert "TestProvider" in str(error)
    assert error.resource_name == "TestProvider"


def test_resource_not_found_error_from_string() -> None:
    """ResourceNotFoundError should accept a string."""
    error = ResourceNotFoundError("custom_resource")

    assert "custom_resource" in str(error)
    assert error.resource_name == "custom_resource"


def test_resource_not_found_inherits_resource_error() -> None:
    """ResourceNotFoundError should inherit from ResourceError."""
    error = ResourceNotFoundError(TestProvider)

    assert isinstance(error, ResourceError)


# =============================================================================
# ResourceRegistry Basic Tests
# =============================================================================


def test_registry_init() -> None:
    """ResourceRegistry should initialize."""
    registry = ResourceRegistry()

    assert registry is not None


def test_registry_register_and_get() -> None:
    """ResourceRegistry should register and retrieve providers."""
    registry = ResourceRegistry()
    provider = TestProvider()

    registry.register(TestProvider, provider)

    retrieved = registry.get(TestProvider)
    assert retrieved is provider


def test_registry_get_resource_value() -> None:
    """Registry providers should return their resource value."""
    registry = ResourceRegistry()
    provider = TestProvider(value="hello", count=TEST_COUNT_42)
    registry.register(TestProvider, provider)

    retrieved = cast("TestProvider", registry.get(TestProvider))
    resource = retrieved.get()

    assert resource.value == "hello"
    assert resource.count == TEST_COUNT_42


def test_registry_get_nonexistent_raises() -> None:
    """ResourceRegistry.get should raise for unknown types."""
    registry = ResourceRegistry()

    with pytest.raises(ResourceNotFoundError, match="TestProvider"):
        registry.get(TestProvider)


def test_registry_duplicate_registration_raises() -> None:
    """ResourceRegistry should reject duplicate registrations."""
    registry = ResourceRegistry()
    provider1 = TestProvider()
    provider2 = TestProvider()

    registry.register(TestProvider, provider1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(TestProvider, provider2)


def test_registry_register_or_replace() -> None:
    """ResourceRegistry.register_or_replace should allow replacement."""
    registry = ResourceRegistry()
    provider1 = TestProvider(value="first")
    provider2 = TestProvider(value="second")

    registry.register(TestProvider, provider1)
    registry.register_or_replace(TestProvider, provider2)

    retrieved = cast("TestProvider", registry.get(TestProvider))
    assert retrieved.get().value == "second"


# =============================================================================
# ResourceRegistry Name-based Access Tests
# =============================================================================


def test_registry_get_by_name() -> None:
    """ResourceRegistry should support name-based lookup."""
    registry = ResourceRegistry()
    provider = TestProvider()

    registry.register(TestProvider, provider)

    retrieved = registry.get_by_name("TestProvider")
    assert retrieved is provider


def test_registry_get_by_name_nonexistent_raises() -> None:
    """ResourceRegistry.get_by_name should raise KeyError for unknown names."""
    registry = ResourceRegistry()

    with pytest.raises(KeyError, match="unknown_name"):
        registry.get_by_name("unknown_name")


# =============================================================================
# ResourceRegistry Has Tests
# =============================================================================


def test_registry_has_registered() -> None:
    """ResourceRegistry.has should return True for registered types."""
    registry = ResourceRegistry()
    provider = TestProvider()
    registry.register(TestProvider, provider)

    assert registry.has(TestProvider) is True


def test_registry_has_unregistered() -> None:
    """ResourceRegistry.has should return False for unregistered types."""
    registry = ResourceRegistry()

    assert registry.has(TestProvider) is False


def test_registry_has_by_name() -> None:
    """ResourceRegistry.has_by_name should check name existence."""
    registry = ResourceRegistry()
    provider = TestProvider()
    registry.register(TestProvider, provider)

    assert registry.has_by_name("TestProvider") is True
    assert registry.has_by_name("unknown") is False


# =============================================================================
# ResourceRegistry Get Or None Tests
# =============================================================================


def test_registry_get_or_none_found() -> None:
    """ResourceRegistry.get_or_none should return provider when found."""
    registry = ResourceRegistry()
    provider = TestProvider()
    registry.register(TestProvider, provider)

    result = registry.get_or_none(TestProvider)

    assert result is provider


def test_registry_get_or_none_not_found() -> None:
    """ResourceRegistry.get_or_none should return None when not found."""
    registry = ResourceRegistry()

    result = registry.get_or_none(TestProvider)

    assert result is None


# =============================================================================
# ResourceRegistry Require Tests
# =============================================================================


def test_registry_require_found() -> None:
    """ResourceRegistry.require should return resource value when found."""
    registry = ResourceRegistry()
    provider = TestProvider(value="test_value")
    registry.register(TestProvider, provider)

    result = registry.require(TestProvider)

    # require returns the resource value, not the provider
    assert isinstance(result, TestResource)
    assert result.value == "test_value"


def test_registry_require_not_found() -> None:
    """ResourceRegistry.require should raise when not found."""
    registry = ResourceRegistry()

    with pytest.raises(ResourceNotFoundError):
        registry.require(TestProvider)


def test_registry_require_by_name_found() -> None:
    """ResourceRegistry.require_by_name should return resource when found."""
    registry = ResourceRegistry()
    provider = TestProvider(value="named_value")
    registry.register(TestProvider, provider)

    result = registry.require_by_name("TestProvider")

    # require_by_name returns the resource value
    assert isinstance(result, TestResource)
    assert result.value == "named_value"


# =============================================================================
# ResourceRegistry Clear Tests
# =============================================================================


def test_registry_clear() -> None:
    """ResourceRegistry.clear should remove all providers."""
    registry = ResourceRegistry()
    registry.register(TestProvider, TestProvider())
    registry.register(ListProvider, ListProvider())

    registry.clear()

    assert registry.has(TestProvider) is False
    assert registry.has(ListProvider) is False


# =============================================================================
# ResourceRegistry Multiple Providers Tests
# =============================================================================


def test_registry_multiple_different_types() -> None:
    """ResourceRegistry should handle multiple provider types."""
    registry = ResourceRegistry()
    test_provider = TestProvider(value="test")
    list_provider = ListProvider(items=["a", "b", "c"])

    registry.register(TestProvider, test_provider)
    registry.register(ListProvider, list_provider)

    assert cast("TestProvider", registry.get(TestProvider)).get().value == "test"
    items = cast("ListProvider", registry.get(ListProvider)).get()
    assert len(items) == EXPECTED_LEN_3


# =============================================================================
# ResourceProvider Tests
# =============================================================================


def test_provider_resource_name_attribute() -> None:
    """ResourceProvider should have RESOURCE_NAME class attribute."""
    provider = TestProvider()

    assert TestProvider.RESOURCE_NAME == "test_resource"
    # Also verify via the instance access
    assert provider.RESOURCE_NAME == "test_resource"


def test_failing_provider_raises() -> None:
    """FailingProvider should raise ResourceError."""
    provider = FailingProvider()

    with pytest.raises(ResourceError, match="unavailable"):
        provider.get()


# =============================================================================
# Integration Tests
# =============================================================================


def test_registry_workflow() -> None:
    """Test typical registry workflow."""
    # Create registry
    registry = ResourceRegistry()

    # Register providers
    test_provider = TestProvider(value="hello", count=TEST_COUNT_100)
    list_provider = ListProvider(items=["x", "y"])

    registry.register(TestProvider, test_provider)
    registry.register(ListProvider, list_provider)

    # Use providers
    test_resource = cast("TestProvider", registry.get(TestProvider)).get()
    list_resource = cast("ListProvider", registry.get(ListProvider)).get()

    assert test_resource.value == "hello"
    assert test_resource.count == TEST_COUNT_100
    assert list(list_resource) == ["x", "y"]


def test_registry_optional_access() -> None:
    """Registry should support optional access pattern."""
    registry = ResourceRegistry()

    # Check before access using has()
    if registry.has(TestProvider):
        provider = cast("TestProvider", registry.get(TestProvider))
        _ = provider.get()
    else:
        # Provider not registered - expected path
        pass

    # Now register and access
    registry.register(TestProvider, TestProvider())
    assert registry.has(TestProvider)


def test_registry_registered_types() -> None:
    """Registry should report registered types."""
    registry = ResourceRegistry()
    registry.register(TestProvider, TestProvider())
    registry.register(ListProvider, ListProvider())

    # registered_types is a property, not a method
    types = registry.registered_types

    assert TestProvider in types
    assert ListProvider in types


# =============================================================================
# ToolsProvider Tests
# =============================================================================


def test_tools_provider_with_pre_configured_service(tmp_path: Path) -> None:
    """ToolsProvider should use pre-configured service if provided."""
    tools_cfg = ToolsConfig.default()
    runner = ToolRunner(cache_dir=tmp_path, tools_config=tools_cfg)
    service = ToolService(runner, tools_cfg)

    provider = ToolsProvider(
        tools_config=tools_cfg,
        cache_dir=tmp_path,
        service=service,
    )

    result = provider.get()
    assert result is service


def test_tools_provider_creates_service_lazily(tmp_path: Path) -> None:
    """ToolsProvider should create service on first access."""
    tools_cfg = ToolsConfig.default()
    provider = ToolsProvider(tools_config=tools_cfg, cache_dir=tmp_path)

    assert provider.is_loaded is False
    service = provider.get()
    assert service is not None
    assert provider.is_loaded is True


def test_tools_provider_runner_property_before_load(tmp_path: Path) -> None:
    """ToolsProvider.runner should return None before loading."""
    tools_cfg = ToolsConfig.default()
    provider = ToolsProvider(tools_config=tools_cfg, cache_dir=tmp_path)

    # No runner configured, not loaded yet
    assert provider.runner is None


def test_tools_provider_runner_property_with_preconfigured(tmp_path: Path) -> None:
    """ToolsProvider.runner should return pre-configured runner."""
    tools_cfg = ToolsConfig.default()
    runner = ToolRunner(cache_dir=tmp_path, tools_config=tools_cfg)

    provider = ToolsProvider(
        tools_config=tools_cfg,
        cache_dir=tmp_path,
        runner=runner,
    )

    assert provider.runner is runner


def test_tools_provider_runner_property_after_load(tmp_path: Path) -> None:
    """ToolsProvider.runner should return runner after loading."""
    tools_cfg = ToolsConfig.default()
    provider = ToolsProvider(tools_config=tools_cfg, cache_dir=tmp_path)

    # Load the service
    _ = provider.get()

    # Now runner should be available from the service
    assert provider.runner is not None


# =============================================================================
# IngestStorageService Tests
# =============================================================================


def test_ingest_storage_service_from_gateway(fresh_gateway: StorageGateway) -> None:
    """IngestStorageService.from_gateway should create service."""
    service = IngestStorageService.from_gateway(fresh_gateway)
    assert service.storage is not None


def test_ingest_storage_service_run_batch(fresh_gateway: StorageGateway) -> None:
    """IngestStorageService.run_batch should write rows."""
    service = IngestStorageService.from_gateway(fresh_gateway)

    rows = [
        ("mod1", "path1.py", "test/repo", "abc123", "python", "[]", "[]"),
    ]
    result = service.run_batch("core.modules", rows, scope="test/repo@abc123")

    assert result.rows_written == 1


def test_ingest_storage_service_run_batch_with_delete(fresh_gateway: StorageGateway) -> None:
    """IngestStorageService.run_batch should delete before write if params given."""
    service = IngestStorageService.from_gateway(fresh_gateway)

    # First insert some data
    rows1 = [
        ("mod1", "path1.py", "test/repo", "abc123", "python", "[]", "[]"),
    ]
    service.run_batch("core.modules", rows1)

    # Now do a batch with delete_params
    rows2 = [
        ("mod2", "path2.py", "test/repo", "abc123", "python", "[]", "[]"),
    ]
    result = service.run_batch(
        "core.modules",
        rows2,
        delete_params=["test/repo", "abc123"],
    )

    assert result.rows_written == 1


# =============================================================================
# TrackerConfig Tests
# =============================================================================


def test_tracker_config_defaults() -> None:
    """TrackerConfig should have sensible defaults."""
    config = TrackerConfig()

    assert config.scratch is None
    assert config.profile is None
    assert config.policy is None
    assert config.full_rebuild is False


def test_tracker_config_with_full_rebuild() -> None:
    """TrackerConfig should accept full_rebuild flag."""
    config = TrackerConfig(full_rebuild=True)

    assert config.full_rebuild is True


def test_tracker_config_with_profile(tmp_path: Path) -> None:
    """TrackerConfig should accept scan profile."""
    profile = ScanProfile(
        repo_root=tmp_path,
        source_roots=(tmp_path,),
        include_globs=("**/*.py",),
    )
    config = TrackerConfig(profile=profile)

    assert config.profile is profile


# =============================================================================
# TrackerProvider Tests
# =============================================================================


def test_tracker_provider_initialization(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """TrackerProvider should initialize with gateway and snapshot."""
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    provider = TrackerProvider(fresh_gateway, snapshot)

    # RESOURCE_NAME ClassVar is "tracker" for consistency with core resources
    assert provider.resource_name == "tracker"
    assert provider.is_loaded is False


def test_tracker_provider_with_config(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """TrackerProvider should accept TrackerConfig."""
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    config = TrackerConfig(full_rebuild=True)
    provider = TrackerProvider(fresh_gateway, snapshot, config)

    # Verify provider was created successfully - configuration behavior is tested
    # through actual tracker behavior in other tests, not by accessing private state
    assert provider is not None
    # RESOURCE_NAME ClassVar is "tracker" for consistency with core resources
    assert provider.resource_name == "tracker"


def test_tracker_provider_get_or_create_alias(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """TrackerProvider.get_or_create should be alias for get."""
    # Create a simple repo structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("# main module\n", encoding="utf-8")

    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    provider = TrackerProvider(fresh_gateway, snapshot)

    # get_or_create should work same as get
    tracker = provider.get_or_create()
    assert tracker is not None
    assert provider.is_loaded is True


def test_tracker_provider_load_creates_tracker(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """TrackerProvider._load should create a fresh tracker."""
    # Create a simple repo structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("# main module\n", encoding="utf-8")

    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    provider = TrackerProvider(fresh_gateway, snapshot)

    tracker = provider.get()
    assert tracker is not None
    assert provider.is_loaded is True


# =============================================================================
# LazyResource Lifecycle Tests
# =============================================================================


class MockLazyResource(LazyResourceBase[str]):
    """Mock lazy resource for testing lifecycle."""

    def __init__(self, value: str, *, should_fail: bool = False) -> None:
        """Initialize the mock resource.

        Parameters
        ----------
        value
            The value to return when loaded.
        should_fail
            If True, raise ValueError on load.
        """
        super().__init__("MockResource")
        self._value = value
        self._should_fail = should_fail

    def _load(self) -> str:
        """Load the mock resource.

        Returns
        -------
        str
            The stored value.

        Raises
        ------
        ValueError
            If should_fail was set to True during initialization.
        """
        if self._should_fail:
            msg = "Mock load failure"
            raise ValueError(msg)
        return self._value


def test_lazy_resource_initial_state() -> None:
    """LazyResource should start unloaded."""
    resource = MockLazyResource("test")

    assert resource.is_loaded is False
    assert resource.resource_name == "MockResource"


def test_lazy_resource_get_loads_value() -> None:
    """LazyResource.get should load and return value."""
    resource = MockLazyResource("hello")

    result = resource.get()

    assert result == "hello"
    assert resource.is_loaded is True


def test_lazy_resource_get_caches_value() -> None:
    """LazyResource.get should cache the loaded value."""
    resource = MockLazyResource("cached")

    # First call loads
    result1 = resource.get()
    # Second call returns cached
    result2 = resource.get()

    assert result1 == result2
    assert result1 == "cached"


def test_lazy_resource_get_or_none_success() -> None:
    """LazyResource.get_or_none should return value on success."""
    resource = MockLazyResource("value")

    result = resource.get_or_none()

    assert result == "value"


def test_lazy_resource_get_or_none_failure() -> None:
    """LazyResource.get_or_none should return None on failure."""
    resource = MockLazyResource("value", should_fail=True)

    result = resource.get_or_none()

    assert result is None


def test_lazy_resource_invalidate() -> None:
    """LazyResource.invalidate should reset state."""
    resource = MockLazyResource("test")

    # Load the resource
    resource.get()
    assert resource.is_loaded is True

    # Invalidate
    resource.invalidate()
    assert resource.is_loaded is False


def test_lazy_resource_set_preloaded() -> None:
    """LazyResource.set_preloaded should set value without loading."""
    resource = MockLazyResource("will_not_load")

    resource.set_preloaded("preloaded_value")

    assert resource.is_loaded is True
    assert resource.get() == "preloaded_value"


def test_lazy_resource_error_handling() -> None:
    """LazyResource should handle load errors."""
    resource = MockLazyResource("value", should_fail=True)

    with pytest.raises(ResourceNotLoadedError) as exc_info:
        resource.get()

    assert "MockResource" in str(exc_info.value)


def test_lazy_resource_error_cached() -> None:
    """LazyResource should cache load errors."""
    resource = MockLazyResource("value", should_fail=True)

    # First call fails
    with pytest.raises(ResourceNotLoadedError):
        resource.get()

    # Second call should also fail with cached error
    with pytest.raises(ResourceNotLoadedError):
        resource.get()


# =============================================================================
# ResourceNotLoadedError Tests
# =============================================================================


def test_resource_not_loaded_error_basic() -> None:
    """ResourceNotLoadedError should format message correctly."""
    error = ResourceNotLoadedError("TestResource")

    assert "TestResource" in str(error)
    assert error.resource_type == "TestResource"
    assert error.reason is None


def test_resource_not_loaded_error_with_reason() -> None:
    """ResourceNotLoadedError should include reason in message."""
    error = ResourceNotLoadedError("TestResource", "file not found")

    assert "TestResource" in str(error)
    assert "file not found" in str(error)
    assert error.reason == "file not found"
