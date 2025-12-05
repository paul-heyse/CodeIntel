"""Test decorator factory from codeintel.core.plugins.decorators.

This module tests:
- make_plugin_instance() with metadata conversion
- Registration callback invocation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.plugins.decorators import make_plugin_instance
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.core.plugins.types.result import PluginResult

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Test Types
# =============================================================================


@dataclass
class MockContext:
    """Mock execution context."""

    value: str


@dataclass
class MockOptions:
    """Mock options for testing."""

    name: str
    version: str = "1.0.0"


@dataclass
class MockPlugin:
    """Mock plugin for testing."""

    metadata: PluginMetadata
    execute_fn: Callable[[MockContext], PluginResult]


# =============================================================================
# make_plugin_instance Tests
# =============================================================================


def test_make_plugin_instance_creates_plugin() -> None:
    """Verify make_plugin_instance creates a plugin."""

    def execute_fn(_ctx: MockContext) -> PluginResult:
        return PluginResult.ok()

    def to_metadata(opts: MockOptions, _fn: Callable) -> PluginMetadata:
        return PluginMetadata(
            name=opts.name,
            description="test",
            kind="analytics",
            stage="function",
            version=opts.version,
        )

    def plugin_factory(
        meta: PluginMetadata,
        fn: Callable[[MockContext], PluginResult],
    ) -> MockPlugin:
        return MockPlugin(metadata=meta, execute_fn=fn)

    options = MockOptions(name="test.plugin", version="2.0.0")

    plugin = make_plugin_instance(
        fn=execute_fn,
        options=options,
        plugin_factory=plugin_factory,
        to_metadata=to_metadata,
    )

    assert isinstance(plugin, MockPlugin)
    assert plugin.metadata.name == "test.plugin"
    assert plugin.metadata.version == "2.0.0"


def test_make_plugin_instance_calls_register() -> None:
    """Verify make_plugin_instance calls registration callback."""
    registered: list[MockPlugin] = []

    def execute_fn(_ctx: MockContext) -> PluginResult:
        return PluginResult.ok()

    def to_metadata(opts: MockOptions, _fn: Callable) -> PluginMetadata:
        return PluginMetadata(
            name=opts.name,
            description="test",
            kind="analytics",
            stage="function",
        )

    def plugin_factory(
        meta: PluginMetadata,
        fn: Callable[[MockContext], PluginResult],
    ) -> MockPlugin:
        return MockPlugin(metadata=meta, execute_fn=fn)

    def register_fn(plugin: MockPlugin) -> None:
        registered.append(plugin)

    options = MockOptions(name="registered.plugin")

    plugin = make_plugin_instance(
        fn=execute_fn,
        options=options,
        plugin_factory=plugin_factory,
        to_metadata=to_metadata,
        register_fn=register_fn,
    )

    assert len(registered) == 1
    assert registered[0] is plugin


def test_make_plugin_instance_no_register() -> None:
    """Verify make_plugin_instance works without registration callback."""

    def execute_fn(_ctx: MockContext) -> PluginResult:
        return PluginResult.ok()

    def to_metadata(opts: MockOptions, _fn: Callable) -> PluginMetadata:
        return PluginMetadata(
            name=opts.name,
            description="test",
            kind="analytics",
            stage="function",
        )

    def plugin_factory(
        meta: PluginMetadata,
        fn: Callable[[MockContext], PluginResult],
    ) -> MockPlugin:
        return MockPlugin(metadata=meta, execute_fn=fn)

    options = MockOptions(name="no.register")

    # Should not raise when register_fn is None
    plugin = make_plugin_instance(
        fn=execute_fn,
        options=options,
        plugin_factory=plugin_factory,
        to_metadata=to_metadata,
        register_fn=None,
    )

    assert plugin.metadata.name == "no.register"


def test_make_plugin_instance_preserves_execute_fn() -> None:
    """Verify make_plugin_instance preserves the original execute function."""
    call_log: list[str] = []

    def execute_fn(ctx: MockContext) -> PluginResult:
        call_log.append(ctx.value)
        return PluginResult.ok()

    def to_metadata(opts: MockOptions, _fn: Callable) -> PluginMetadata:
        return PluginMetadata(
            name=opts.name,
            description="test",
            kind="analytics",
            stage="function",
        )

    def plugin_factory(
        meta: PluginMetadata,
        fn: Callable[[MockContext], PluginResult],
    ) -> MockPlugin:
        return MockPlugin(metadata=meta, execute_fn=fn)

    options = MockOptions(name="test")

    plugin = make_plugin_instance(
        fn=execute_fn,
        options=options,
        plugin_factory=plugin_factory,
        to_metadata=to_metadata,
    )

    # Call the execute function
    ctx = MockContext(value="test-value")
    plugin.execute_fn(ctx)

    assert call_log == ["test-value"]


def test_make_plugin_instance_passes_fn_to_to_metadata() -> None:
    """Verify make_plugin_instance passes function to to_metadata."""
    received_fns: list[Callable] = []

    def execute_fn(_ctx: MockContext) -> PluginResult:
        return PluginResult.ok()

    def to_metadata(_opts: MockOptions, fn: Callable) -> PluginMetadata:
        received_fns.append(fn)
        return PluginMetadata(
            name=fn.__name__,  # Use function name as plugin name
            description="test",
            kind="analytics",
            stage="function",
        )

    def plugin_factory(
        meta: PluginMetadata,
        fn: Callable[[MockContext], PluginResult],
    ) -> MockPlugin:
        return MockPlugin(metadata=meta, execute_fn=fn)

    options = MockOptions(name="ignored")

    plugin = make_plugin_instance(
        fn=execute_fn,
        options=options,
        plugin_factory=plugin_factory,
        to_metadata=to_metadata,
    )

    assert len(received_fns) == 1
    assert received_fns[0] is execute_fn
    assert plugin.metadata.name == "execute_fn"


def test_make_plugin_instance_uses_custom_types() -> None:
    """Verify make_plugin_instance works with different type parameters."""

    @dataclass
    class CustomContext:
        data: dict[str, int]

    @dataclass
    class CustomOptions:
        plugin_name: str
        stage_name: str

    @dataclass
    class CustomPlugin:
        meta: PluginMetadata
        fn: Callable[[CustomContext], PluginResult]

    def execute_fn(ctx: CustomContext) -> PluginResult:
        return PluginResult.ok(meta={"total": sum(ctx.data.values())})

    def to_metadata(opts: CustomOptions, _fn: Callable) -> PluginMetadata:
        return PluginMetadata(
            name=opts.plugin_name,
            description="custom",
            kind="builder",
            stage="goid",
        )

    def plugin_factory(
        meta: PluginMetadata,
        fn: Callable[[CustomContext], PluginResult],
    ) -> CustomPlugin:
        return CustomPlugin(meta=meta, fn=fn)

    options = CustomOptions(plugin_name="custom.plugin", stage_name="goid")

    plugin = make_plugin_instance(
        fn=execute_fn,
        options=options,
        plugin_factory=plugin_factory,
        to_metadata=to_metadata,
    )

    assert isinstance(plugin, CustomPlugin)
    assert plugin.meta.name == "custom.plugin"
    assert plugin.meta.kind == "builder"
