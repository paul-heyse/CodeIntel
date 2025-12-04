"""Test functional plugin from codeintel.core.plugins.functional.

This module tests:
- BaseFunctionalPlugin metadata property
- execute() delegation to wrapped function
- validate_inputs() with custom and default validators
"""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.core.plugins.functional import BaseFunctionalPlugin
from codeintel.core.plugins.protocol import PluginMetadata, ValidationResult
from codeintel.core.plugins.result import PluginResult

# =============================================================================
# Test Context
# =============================================================================


@dataclass
class MockContext:
    """Mock execution context for testing."""

    value: str


# =============================================================================
# BaseFunctionalPlugin Tests
# =============================================================================


def test_functional_plugin_metadata() -> None:
    """Verify metadata property returns the provided metadata."""
    metadata = PluginMetadata(
        name="test.functional",
        description="Test functional plugin",
        kind="analytics",
        stage="function",
    )

    def execute_fn(_ctx: MockContext) -> PluginResult:
        return PluginResult.ok()

    plugin = BaseFunctionalPlugin(
        _metadata=metadata,
        _execute_fn=execute_fn,
    )

    assert plugin.metadata is metadata
    assert plugin.metadata.name == "test.functional"


def test_functional_plugin_execute() -> None:
    """Verify execute() delegates to wrapped function."""
    call_log: list[str] = []

    def execute_fn(ctx: MockContext) -> PluginResult:
        call_log.append(ctx.value)
        return PluginResult.ok(meta={"received": ctx.value})

    metadata = PluginMetadata(
        name="test",
        description="test",
        kind="analytics",
        stage="function",
    )

    plugin = BaseFunctionalPlugin(
        _metadata=metadata,
        _execute_fn=execute_fn,
    )

    ctx = MockContext(value="test-value")
    result = plugin.execute(ctx)

    assert result.success is True
    assert result.meta == {"received": "test-value"}
    assert call_log == ["test-value"]


def test_functional_plugin_execute_returns_failure() -> None:
    """Verify execute() returns failure from wrapped function."""

    def execute_fn(_ctx: MockContext) -> PluginResult:
        return PluginResult.fail("Something went wrong")

    metadata = PluginMetadata(
        name="test",
        description="test",
        kind="analytics",
        stage="function",
    )

    plugin = BaseFunctionalPlugin(
        _metadata=metadata,
        _execute_fn=execute_fn,
    )

    result = plugin.execute(MockContext(value="x"))

    assert result.success is False
    assert result.error == "Something went wrong"


def test_functional_plugin_validate_default() -> None:
    """Verify validate_inputs() returns success when no custom validator."""
    metadata = PluginMetadata(
        name="test",
        description="test",
        kind="analytics",
        stage="function",
    )

    plugin = BaseFunctionalPlugin(
        _metadata=metadata,
        _execute_fn=lambda _: PluginResult.ok(),
        _validate_fn=None,
    )

    result = plugin.validate_inputs(MockContext(value="x"))

    assert result.valid is True


def test_functional_plugin_validate_custom_success() -> None:
    """Verify validate_inputs() uses custom validator when provided."""
    call_log: list[str] = []

    def validate_fn(ctx: MockContext) -> ValidationResult:
        call_log.append(ctx.value)
        return ValidationResult.success()

    metadata = PluginMetadata(
        name="test",
        description="test",
        kind="analytics",
        stage="function",
    )

    plugin = BaseFunctionalPlugin(
        _metadata=metadata,
        _execute_fn=lambda _: PluginResult.ok(),
        _validate_fn=validate_fn,
    )

    ctx = MockContext(value="validated")
    result = plugin.validate_inputs(ctx)

    assert result.valid is True
    assert call_log == ["validated"]


def test_functional_plugin_validate_custom_failure() -> None:
    """Verify validate_inputs() returns failure from custom validator."""

    def validate_fn(_ctx: MockContext) -> ValidationResult:
        return ValidationResult.failure(("Missing required config",))

    metadata = PluginMetadata(
        name="test",
        description="test",
        kind="analytics",
        stage="function",
    )

    plugin = BaseFunctionalPlugin(
        _metadata=metadata,
        _execute_fn=lambda _: PluginResult.ok(),
        _validate_fn=validate_fn,
    )

    result = plugin.validate_inputs(MockContext(value="x"))

    assert result.valid is False
    assert "Missing required config" in result.errors


def test_functional_plugin_with_generic_types() -> None:
    """Verify BaseFunctionalPlugin works with type parameters."""
    # This tests that the generic type parameters work correctly
    plugin: BaseFunctionalPlugin[MockContext, PluginMetadata] = BaseFunctionalPlugin(
        _metadata=PluginMetadata(
            name="typed",
            description="Typed plugin",
            kind="analytics",
            stage="other",
        ),
        _execute_fn=lambda ctx: PluginResult.ok(meta={"ctx": ctx.value}),
    )

    result = plugin.execute(MockContext(value="typed-test"))

    assert result.success
    assert result.meta["ctx"] == "typed-test"


def test_functional_plugin_dataclass_fields() -> None:
    """Verify BaseFunctionalPlugin dataclass fields are accessible."""
    metadata = PluginMetadata(
        name="test",
        description="test",
        kind="analytics",
        stage="function",
    )

    def execute_fn(_ctx: MockContext) -> PluginResult:
        return PluginResult.ok()

    def validate_fn(_ctx: MockContext) -> ValidationResult:
        return ValidationResult.success()

    plugin = BaseFunctionalPlugin(
        _metadata=metadata,
        _execute_fn=execute_fn,
        _validate_fn=validate_fn,
    )

    # Dataclass fields should be directly accessible
    assert plugin._metadata is metadata  # noqa: SLF001
    assert plugin._execute_fn is execute_fn  # noqa: SLF001
    assert plugin._validate_fn is validate_fn  # noqa: SLF001


def test_functional_plugin_execute_passes_context() -> None:
    """Verify execute() passes context to wrapped function."""
    received_contexts: list[MockContext] = []

    def execute_fn(ctx: MockContext) -> PluginResult:
        received_contexts.append(ctx)
        return PluginResult.ok()

    metadata = PluginMetadata(
        name="test",
        description="test",
        kind="analytics",
        stage="function",
    )

    plugin = BaseFunctionalPlugin(
        _metadata=metadata,
        _execute_fn=execute_fn,
    )

    ctx = MockContext(value="unique-value")
    plugin.execute(ctx)

    assert len(received_contexts) == 1
    assert received_contexts[0] is ctx
    assert received_contexts[0].value == "unique-value"
