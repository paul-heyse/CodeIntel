"""Test plugin execution context from codeintel.core.plugins.context.

This module tests:
- PluginScratch operations (declare, consume, has, cleanup, keys, len)
- ConfigProvider operations (get, get_optional, has, register)
- PluginExecutionContext properties and methods
- PluginExecutionContextBuilder fluent API

Note: Uses shared core fixtures from core/conftest.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.execution import RunContext
from codeintel.core.plugins.execution.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.core.resources.registry import ResourceRegistry
from codeintel.storage.gateway import StorageGateway
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.fakes import (
    create_graph_gateway,
    create_test_run_context,
    create_test_snapshot,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class SampleConfig:
    """Sample configuration class for testing."""

    value: str


@dataclass
class AnotherConfig:
    """Another sample configuration class for testing."""

    number: int


class SampleResource:
    """Sample resource class for testing."""


def make_test_snapshot(tmp_dir: Path | None = None) -> SnapshotRef:
    """Create a real SnapshotRef for testing.

    Parameters
    ----------
    tmp_dir
        Optional directory for repo_root. Defaults to a mock path.

    Returns
    -------
    SnapshotRef
        Test snapshot reference.
    """
    return create_test_snapshot(tmp_dir)


def make_test_gateway() -> StorageGateway:
    """Create a real in-memory StorageGateway for testing.

    Returns
    -------
    StorageGateway
        In-memory storage gateway with schema applied.
    """
    return create_graph_gateway()


def make_test_run_context(snapshot: SnapshotRef, run_id: str = "test-run-id") -> RunContext:
    """Create a real RunContext for testing.

    Delegates to the centralized create_test_run_context helper.

    Parameters
    ----------
    snapshot
        Snapshot reference.
    run_id
        Run identifier.

    Returns
    -------
    RunContext
        Test run context.
    """
    return create_test_run_context(snapshot, run_id=run_id)


def make_test_build_paths(tmp_dir: Path) -> BuildPaths:
    """Create a real BuildPaths for testing.

    Parameters
    ----------
    tmp_dir
        Temporary directory for paths.

    Returns
    -------
    BuildPaths
        Test build paths.
    """
    return BuildPaths.from_repo_root(tmp_dir, build_dir=tmp_dir / "build")


@pytest.fixture
def test_gateway(core_gateway: StorageGateway) -> StorageGateway:
    """Alias for core_gateway for backward compatibility.

    Parameters
    ----------
    core_gateway
        Shared core gateway fixture.

    Returns
    -------
    StorageGateway
        In-memory gateway with schema applied.
    """
    return core_gateway


@pytest.fixture
def test_snapshot(core_snapshot: SnapshotRef) -> SnapshotRef:
    """Alias for core_snapshot for backward compatibility.

    Parameters
    ----------
    core_snapshot
        Shared core snapshot fixture.

    Returns
    -------
    SnapshotRef
        Test snapshot reference.
    """
    return core_snapshot


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
    expected_count = len({"key1", "key2"})
    assert len(scratch) == expected_count


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

    assert not provider.has(SampleConfig)


def test_config_provider_with_initial_configs() -> None:
    """Verify ConfigProvider accepts initial configs."""
    config = SampleConfig(value="test")
    provider = ConfigProvider({SampleConfig: config})

    assert provider.has(SampleConfig)
    assert provider.get(SampleConfig) is config


def test_config_provider_get() -> None:
    """Verify get returns registered config."""
    provider = ConfigProvider()
    config = SampleConfig(value="test")
    provider.register(SampleConfig, config)

    result = provider.get(SampleConfig)

    assert result is config


def test_config_provider_get_missing() -> None:
    """Verify get raises ValueError for missing config."""
    provider = ConfigProvider()

    with pytest.raises(ValueError, match="SampleConfig not available"):
        provider.get(SampleConfig)


def test_config_provider_get_optional() -> None:
    """Verify get_optional returns config or None."""
    provider = ConfigProvider()
    config = SampleConfig(value="test")
    provider.register(SampleConfig, config)

    assert provider.get_optional(SampleConfig) is config
    assert provider.get_optional(AnotherConfig) is None


def test_config_provider_has() -> None:
    """Verify has returns correct boolean."""
    provider = ConfigProvider()
    provider.register(SampleConfig, SampleConfig(value="test"))

    assert provider.has(SampleConfig) is True
    assert provider.has(AnotherConfig) is False


def test_config_provider_register() -> None:
    """Verify register adds config."""
    provider = ConfigProvider()

    provider.register(SampleConfig, SampleConfig(value="test"))

    assert provider.has(SampleConfig)


# =============================================================================
# PluginExecutionContext Tests
# =============================================================================


def test_plugin_execution_context_properties(tmp_path: Path) -> None:
    """Verify context properties return snapshot data."""
    gateway = make_test_gateway()
    try:
        snapshot = make_test_snapshot(tmp_path)

        ctx = PluginExecutionContext(
            gateway=gateway,
            snapshot=snapshot,
            run_id="run-123",
        )

        assert ctx.repo == DEFAULT_REPO
        assert ctx.commit == DEFAULT_COMMIT
        assert ctx.repo_root == tmp_path
    finally:
        gateway.close()


def test_plugin_execution_context_effective_run_id_from_context(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify effective_run_id prefers run_context."""
    run_context = make_test_run_context(test_snapshot, run_id="unified-run-id")

    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
        run_id="fallback-run-id",
        run_context=run_context,
    )

    assert ctx.effective_run_id == "unified-run-id"


def test_plugin_execution_context_effective_run_id_fallback(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify effective_run_id falls back to run_id."""
    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
        run_id="fallback-run-id",
    )

    assert ctx.effective_run_id == "fallback-run-id"


def test_plugin_execution_context_get_config(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify get_config delegates to ConfigProvider."""
    config = SampleConfig(value="test")
    configs = ConfigProvider({SampleConfig: config})

    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
        configs=configs,
    )

    assert ctx.get_config(SampleConfig) is config


def test_plugin_execution_context_get_optional_config(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify get_optional_config delegates to ConfigProvider."""
    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
    )

    assert ctx.get_optional_config(SampleConfig) is None


def test_plugin_execution_context_has_config(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify has_config delegates to ConfigProvider."""
    configs = ConfigProvider({SampleConfig: SampleConfig(value="test")})

    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
        configs=configs,
    )

    assert ctx.has_config(SampleConfig) is True
    assert ctx.has_config(AnotherConfig) is False


def test_plugin_execution_context_require_resource(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify require delegates to ResourceRegistry."""
    resources = ResourceRegistry()
    resource = SampleResource()
    resources.register(SampleResource, resource)

    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
        resources=resources,
    )

    assert ctx.require(SampleResource) is resource


def test_plugin_execution_context_require_or_none(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify require_or_none returns None for missing."""
    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
    )

    assert ctx.require_or_none(SampleResource) is None


def test_plugin_execution_context_has_resource(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify has_resource checks registry."""
    resources = ResourceRegistry()
    resources.register(SampleResource, SampleResource())

    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
        resources=resources,
    )

    assert ctx.has_resource(SampleResource) is True


def test_plugin_execution_context_require_by_name(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify require_by_name uses string lookup."""
    resources = ResourceRegistry()
    resource = SampleResource()
    resources.register_by_name("SampleResource", resource)

    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
        resources=resources,
    )

    assert ctx.require_by_name("SampleResource") is resource


def test_plugin_execution_context_has_resource_by_name(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify has_resource_by_name checks by string."""
    resources = ResourceRegistry()
    resources.register_by_name("MyResource", SampleResource())

    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
        resources=resources,
    )

    assert ctx.has_resource_by_name("MyResource") is True
    assert ctx.has_resource_by_name("Missing") is False


def test_plugin_execution_context_register_resource(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify register_resource adds to registry."""
    ctx = PluginExecutionContext(
        gateway=test_gateway,
        snapshot=test_snapshot,
    )
    resource = SampleResource()

    ctx.register_resource(SampleResource, resource)

    assert ctx.has_resource(SampleResource) is True
    assert ctx.require(SampleResource) is resource


# =============================================================================
# PluginExecutionContextBuilder Tests
# =============================================================================


def test_builder_basic(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify builder creates basic context."""
    builder = PluginExecutionContextBuilder(
        gateway=test_gateway,
        snapshot=test_snapshot,
        run_id="run-123",
    )
    ctx = builder.build()

    assert ctx.gateway is test_gateway
    assert ctx.snapshot is test_snapshot
    assert ctx.run_id == "run-123"


def test_builder_with_config(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify builder adds configs."""
    config = SampleConfig(value="builder")

    ctx = (
        PluginExecutionContextBuilder(test_gateway, test_snapshot, "run")
        .with_config(SampleConfig, config)
        .build()
    )

    assert ctx.get_config(SampleConfig) is config


def test_builder_with_paths(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
    tmp_path: Path,
) -> None:
    """Verify builder sets paths."""
    paths = make_test_build_paths(tmp_path)

    ctx = (
        PluginExecutionContextBuilder(test_gateway, test_snapshot, "run").with_paths(paths).build()
    )

    assert ctx.paths is paths


def test_builder_with_options(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify builder sets options."""
    options = {"key": "value"}

    ctx = (
        PluginExecutionContextBuilder(test_gateway, test_snapshot, "run")
        .with_options(options)
        .build()
    )

    assert ctx.options == options


def test_builder_with_plugin_name(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify builder sets plugin name."""
    ctx = (
        PluginExecutionContextBuilder(test_gateway, test_snapshot, "run")
        .with_plugin_name("my.plugin")
        .build()
    )

    assert ctx.plugin_name == "my.plugin"


def test_builder_with_extra(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify builder adds extra metadata."""
    ctx = (
        PluginExecutionContextBuilder(test_gateway, test_snapshot, "run")
        .with_extra("meta_key", "meta_value")
        .build()
    )

    assert ctx.extra["meta_key"] == "meta_value"


def test_builder_with_run_context(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify builder sets run context."""
    run_context = make_test_run_context(test_snapshot, run_id="unified-id")

    ctx = (
        PluginExecutionContextBuilder(test_gateway, test_snapshot, "run")
        .with_run_context(run_context)
        .build()
    )

    assert ctx.run_context is run_context
    assert ctx.effective_run_id == "unified-id"


def test_builder_with_resource(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify builder adds resources."""
    resource = SampleResource()

    ctx = (
        PluginExecutionContextBuilder(test_gateway, test_snapshot, "run")
        .with_resource(SampleResource, resource)
        .build()
    )

    assert ctx.require(SampleResource) is resource


def test_builder_with_resources(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify builder sets entire registry."""
    resources = ResourceRegistry()
    resource = SampleResource()
    resources.register(SampleResource, resource)

    ctx = (
        PluginExecutionContextBuilder(test_gateway, test_snapshot, "run")
        .with_resources(resources)
        .build()
    )

    assert ctx.require(SampleResource) is resource


def test_builder_with_scratch(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify builder accepts shared scratch."""
    shared_scratch = PluginScratch()
    shared_scratch.declare("shared_key", "shared_value")

    ctx = PluginExecutionContextBuilder(test_gateway, test_snapshot, "run").build(
        scratch=shared_scratch
    )

    assert ctx.scratch is shared_scratch
    assert ctx.scratch.consume("shared_key") == "shared_value"


def test_builder_fluent_chaining(
    test_gateway: StorageGateway,
    test_snapshot: SnapshotRef,
) -> None:
    """Verify builder supports full fluent chaining."""
    config = SampleConfig(value="chained")
    resource = SampleResource()

    ctx = (
        PluginExecutionContextBuilder(test_gateway, test_snapshot, "run")
        .with_config(SampleConfig, config)
        .with_resource(SampleResource, resource)
        .with_plugin_name("chained.plugin")
        .with_extra("key1", "val1")
        .with_options({"opt": 1})
        .build()
    )

    assert ctx.get_config(SampleConfig) is config
    assert ctx.require(SampleResource) is resource
    assert ctx.plugin_name == "chained.plugin"
    assert ctx.extra["key1"] == "val1"
    assert ctx.options == {"opt": 1}
