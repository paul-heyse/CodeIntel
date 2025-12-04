"""Test plugin execution context from codeintel.core.plugins.context.

This module tests:
- PluginScratch operations (declare, consume, has, cleanup, keys, len)
- ConfigProvider operations (get, get_optional, has, register)
- PluginExecutionContext properties and methods
- PluginExecutionContextBuilder fluent API
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from codeintel.core.plugins.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.core.resources.registry import ResourceRegistry

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class TestConfig:
    """Test configuration class."""

    value: str


@dataclass
class AnotherConfig:
    """Another test configuration class."""

    number: int


class TestResource:
    """Test resource class."""


def make_mock_snapshot() -> SnapshotRef:
    """Create a mock SnapshotRef for testing."""
    snapshot = MagicMock()
    snapshot.repo = "test/repo"
    snapshot.commit = "abc123def"
    snapshot.repo_root = Path("/tmp/test-repo")
    return snapshot


def make_mock_gateway() -> StorageGateway:
    """Create a mock StorageGateway for testing."""
    return MagicMock()


# =============================================================================
# PluginScratch Tests
# =============================================================================


def test_plugin_scratch_declare_and_consume() -> None:
    """Verify declare and consume work correctly."""
    scratch = PluginScratch()

    scratch.declare("key1", {"data": 42})

    result = scratch.consume("key1")
    assert result == {"data": 42}


def test_plugin_scratch_consume_missing_default() -> None:
    """Verify consume returns default for missing key."""
    scratch = PluginScratch()

    result = scratch.consume("missing", "default_value")

    assert result == "default_value"


def test_plugin_scratch_consume_missing_none() -> None:
    """Verify consume returns None for missing key without default."""
    scratch = PluginScratch()

    result = scratch.consume("missing")

    assert result is None


def test_plugin_scratch_has_true() -> None:
    """Verify has returns True for existing key."""
    scratch = PluginScratch()
    scratch.declare("exists", "value")

    assert scratch.has("exists") is True


def test_plugin_scratch_has_false() -> None:
    """Verify has returns False for missing key."""
    scratch = PluginScratch()

    assert scratch.has("missing") is False


def test_plugin_scratch_keys() -> None:
    """Verify keys returns all declared keys."""
    scratch = PluginScratch()
    scratch.declare("key1", "value1")
    scratch.declare("key2", "value2")

    keys = scratch.keys()

    assert set(keys) == {"key1", "key2"}


def test_plugin_scratch_len() -> None:
    """Verify len returns correct count."""
    scratch = PluginScratch()
    assert len(scratch) == 0

    scratch.declare("key1", "value1")
    assert len(scratch) == 1

    scratch.declare("key2", "value2")
    assert len(scratch) == 2


def test_plugin_scratch_register_cleanup() -> None:
    """Verify cleanup callbacks are executed."""
    scratch = PluginScratch()
    cleanup_called = []

    def callback1() -> None:
        cleanup_called.append(1)

    def callback2() -> None:
        cleanup_called.append(2)

    scratch.register_cleanup(callback1)
    scratch.register_cleanup(callback2)

    scratch.cleanup()

    # Should be called in reverse order
    assert cleanup_called == [2, 1]


def test_plugin_scratch_cleanup_clears_store() -> None:
    """Verify cleanup clears the store."""
    scratch = PluginScratch()
    scratch.declare("key", "value")
    assert len(scratch) == 1

    scratch.cleanup()

    assert len(scratch) == 0
    assert not scratch.has("key")


def test_plugin_scratch_cleanup_handles_errors() -> None:
    """Verify cleanup handles errors gracefully."""
    scratch = PluginScratch()
    cleanup_called = []

    def bad_callback() -> None:
        cleanup_called.append("bad")
        msg = "Cleanup error"
        raise RuntimeError(msg)

    def good_callback() -> None:
        cleanup_called.append("good")

    scratch.register_cleanup(good_callback)
    scratch.register_cleanup(bad_callback)

    # Should not raise
    scratch.cleanup()

    # Both should be called (bad first due to reverse order)
    assert cleanup_called == ["bad", "good"]


# =============================================================================
# ConfigProvider Tests
# =============================================================================


def test_config_provider_empty() -> None:
    """Verify empty ConfigProvider has no configs."""
    provider = ConfigProvider()

    assert not provider.has(TestConfig)


def test_config_provider_with_initial_configs() -> None:
    """Verify ConfigProvider accepts initial configs."""
    config = TestConfig(value="test")
    provider = ConfigProvider({TestConfig: config})

    assert provider.has(TestConfig)
    assert provider.get(TestConfig) is config


def test_config_provider_get() -> None:
    """Verify get returns registered config."""
    provider = ConfigProvider()
    config = TestConfig(value="test")
    provider.register(TestConfig, config)

    result = provider.get(TestConfig)

    assert result is config


def test_config_provider_get_missing() -> None:
    """Verify get raises ValueError for missing config."""
    provider = ConfigProvider()

    with pytest.raises(ValueError, match="TestConfig not available"):
        provider.get(TestConfig)


def test_config_provider_get_optional() -> None:
    """Verify get_optional returns config or None."""
    provider = ConfigProvider()
    config = TestConfig(value="test")
    provider.register(TestConfig, config)

    assert provider.get_optional(TestConfig) is config
    assert provider.get_optional(AnotherConfig) is None


def test_config_provider_has() -> None:
    """Verify has returns correct boolean."""
    provider = ConfigProvider()
    provider.register(TestConfig, TestConfig(value="test"))

    assert provider.has(TestConfig) is True
    assert provider.has(AnotherConfig) is False


def test_config_provider_register() -> None:
    """Verify register adds config."""
    provider = ConfigProvider()

    provider.register(TestConfig, TestConfig(value="test"))

    assert provider.has(TestConfig)


# =============================================================================
# PluginExecutionContext Tests
# =============================================================================


def test_plugin_execution_context_properties() -> None:
    """Verify context properties return snapshot data."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        run_id="run-123",
    )

    assert ctx.repo == "test/repo"
    assert ctx.commit == "abc123def"
    assert ctx.repo_root == Path("/tmp/test-repo")


def test_plugin_execution_context_effective_run_id_from_context() -> None:
    """Verify effective_run_id prefers run_context."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    run_context = MagicMock()
    run_context.run_id = "unified-run-id"

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        run_id="fallback-run-id",
        run_context=run_context,
    )

    assert ctx.effective_run_id == "unified-run-id"


def test_plugin_execution_context_effective_run_id_fallback() -> None:
    """Verify effective_run_id falls back to run_id."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        run_id="fallback-run-id",
    )

    assert ctx.effective_run_id == "fallback-run-id"


def test_plugin_execution_context_get_config() -> None:
    """Verify get_config delegates to ConfigProvider."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    config = TestConfig(value="test")
    configs = ConfigProvider({TestConfig: config})

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        configs=configs,
    )

    assert ctx.get_config(TestConfig) is config


def test_plugin_execution_context_get_optional_config() -> None:
    """Verify get_optional_config delegates to ConfigProvider."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
    )

    assert ctx.get_optional_config(TestConfig) is None


def test_plugin_execution_context_has_config() -> None:
    """Verify has_config delegates to ConfigProvider."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    configs = ConfigProvider({TestConfig: TestConfig(value="test")})

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        configs=configs,
    )

    assert ctx.has_config(TestConfig) is True
    assert ctx.has_config(AnotherConfig) is False


def test_plugin_execution_context_require_resource() -> None:
    """Verify require delegates to ResourceRegistry."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    resources = ResourceRegistry()
    resource = TestResource()
    resources.register(TestResource, resource)

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        resources=resources,
    )

    assert ctx.require(TestResource) is resource


def test_plugin_execution_context_require_or_none() -> None:
    """Verify require_or_none returns None for missing."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
    )

    assert ctx.require_or_none(TestResource) is None


def test_plugin_execution_context_has_resource() -> None:
    """Verify has_resource checks registry."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    resources = ResourceRegistry()
    resources.register(TestResource, TestResource())

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        resources=resources,
    )

    assert ctx.has_resource(TestResource) is True


def test_plugin_execution_context_require_by_name() -> None:
    """Verify require_by_name uses string lookup."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    resources = ResourceRegistry()
    resource = TestResource()
    resources.register_by_name("TestResource", resource)

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        resources=resources,
    )

    assert ctx.require_by_name("TestResource") is resource


def test_plugin_execution_context_has_resource_by_name() -> None:
    """Verify has_resource_by_name checks by string."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    resources = ResourceRegistry()
    resources.register_by_name("MyResource", TestResource())

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        resources=resources,
    )

    assert ctx.has_resource_by_name("MyResource") is True
    assert ctx.has_resource_by_name("Missing") is False


def test_plugin_execution_context_register_resource() -> None:
    """Verify register_resource adds to registry."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()

    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
    )
    resource = TestResource()

    ctx.register_resource(TestResource, resource)

    assert ctx.has_resource(TestResource) is True
    assert ctx.require(TestResource) is resource


# =============================================================================
# PluginExecutionContextBuilder Tests
# =============================================================================


def test_builder_basic() -> None:
    """Verify builder creates basic context."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()

    builder = PluginExecutionContextBuilder(
        gateway=gateway,
        snapshot=snapshot,
        run_id="run-123",
    )
    ctx = builder.build()

    assert ctx.gateway is gateway
    assert ctx.snapshot is snapshot
    assert ctx.run_id == "run-123"


def test_builder_with_config() -> None:
    """Verify builder adds configs."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    config = TestConfig(value="builder")

    ctx = (
        PluginExecutionContextBuilder(gateway, snapshot, "run")
        .with_config(TestConfig, config)
        .build()
    )

    assert ctx.get_config(TestConfig) is config


def test_builder_with_paths() -> None:
    """Verify builder sets paths."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    paths = MagicMock()

    ctx = (
        PluginExecutionContextBuilder(gateway, snapshot, "run")
        .with_paths(paths)
        .build()
    )

    assert ctx.paths is paths


def test_builder_with_options() -> None:
    """Verify builder sets options."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    options = {"key": "value"}

    ctx = (
        PluginExecutionContextBuilder(gateway, snapshot, "run")
        .with_options(options)
        .build()
    )

    assert ctx.options == options


def test_builder_with_plugin_name() -> None:
    """Verify builder sets plugin name."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()

    ctx = (
        PluginExecutionContextBuilder(gateway, snapshot, "run")
        .with_plugin_name("my.plugin")
        .build()
    )

    assert ctx.plugin_name == "my.plugin"


def test_builder_with_extra() -> None:
    """Verify builder adds extra metadata."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()

    ctx = (
        PluginExecutionContextBuilder(gateway, snapshot, "run")
        .with_extra("meta_key", "meta_value")
        .build()
    )

    assert ctx.extra["meta_key"] == "meta_value"


def test_builder_with_run_context() -> None:
    """Verify builder sets run context."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    run_context = MagicMock()
    run_context.run_id = "unified-id"

    ctx = (
        PluginExecutionContextBuilder(gateway, snapshot, "run")
        .with_run_context(run_context)
        .build()
    )

    assert ctx.run_context is run_context
    assert ctx.effective_run_id == "unified-id"


def test_builder_with_resource() -> None:
    """Verify builder adds resources."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    resource = TestResource()

    ctx = (
        PluginExecutionContextBuilder(gateway, snapshot, "run")
        .with_resource(TestResource, resource)
        .build()
    )

    assert ctx.require(TestResource) is resource


def test_builder_with_resources() -> None:
    """Verify builder sets entire registry."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    resources = ResourceRegistry()
    resource = TestResource()
    resources.register(TestResource, resource)

    ctx = (
        PluginExecutionContextBuilder(gateway, snapshot, "run")
        .with_resources(resources)
        .build()
    )

    assert ctx.require(TestResource) is resource


def test_builder_with_scratch() -> None:
    """Verify builder accepts shared scratch."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    shared_scratch = PluginScratch()
    shared_scratch.declare("shared_key", "shared_value")

    ctx = (
        PluginExecutionContextBuilder(gateway, snapshot, "run")
        .build(scratch=shared_scratch)
    )

    assert ctx.scratch is shared_scratch
    assert ctx.scratch.consume("shared_key") == "shared_value"


def test_builder_fluent_chaining() -> None:
    """Verify builder supports full fluent chaining."""
    gateway = make_mock_gateway()
    snapshot = make_mock_snapshot()
    config = TestConfig(value="chained")
    resource = TestResource()

    ctx = (
        PluginExecutionContextBuilder(gateway, snapshot, "run")
        .with_config(TestConfig, config)
        .with_resource(TestResource, resource)
        .with_plugin_name("chained.plugin")
        .with_extra("key1", "val1")
        .with_options({"opt": 1})
        .build()
    )

    assert ctx.get_config(TestConfig) is config
    assert ctx.require(TestResource) is resource
    assert ctx.plugin_name == "chained.plugin"
    assert ctx.extra["key1"] == "val1"
    assert ctx.options == {"opt": 1}
