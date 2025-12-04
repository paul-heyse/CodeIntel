"""Test metadata options from codeintel.core.plugins.meta_options.

This module tests:
- BasePluginMetaOptions defaults
- validate_option_keys() with unknown keys
- to_base_metadata() derivation from function
"""

from __future__ import annotations

import pytest

from codeintel.core.plugins.meta_options import (
    BasePluginMetaOptions,
    BasePluginMetaOptionsInput,
)
from codeintel.core.plugins.protocol import PluginInputSpec, PluginResourceHints
from codeintel.core.plugins.result import PluginResult

# =============================================================================
# BasePluginMetaOptions Default Tests
# =============================================================================


def test_base_meta_options_defaults() -> None:
    """Verify BasePluginMetaOptions has sensible defaults."""
    options = BasePluginMetaOptions()

    assert options.name is None
    assert options.description is None
    assert options.kind is None
    assert options.stage is None
    assert options.version == "1.0.0"
    assert options.enabled_by_default is True
    assert options.severity == "fatal"
    assert options.inputs == ()
    assert options.outputs == ()
    assert options.provides == ()
    assert options.requires == ()
    assert options.depends_on == ()
    assert options.resource_hints is None
    assert options.requires_isolation is False
    assert options.isolation_kind == "none"
    assert options.tags == ()


def test_base_meta_options_with_values() -> None:
    """Verify BasePluginMetaOptions accepts custom values."""
    hints = PluginResourceHints(max_runtime_ms=5000)
    input_spec = PluginInputSpec(name="config", type_ref="Config")

    options = BasePluginMetaOptions(
        name="custom.plugin",
        description="A custom plugin",
        kind="builder",
        stage="goid",
        version="2.0.0",
        enabled_by_default=False,
        severity="soft_fail",
        inputs=[input_spec],
        provides=["capability"],
        requires=["dependency"],
        depends_on=["other.plugin"],
        resource_hints=hints,
        requires_isolation=True,
        isolation_kind="process",
        tags=["tag1", "tag2"],
    )

    assert options.name == "custom.plugin"
    assert options.description == "A custom plugin"
    assert options.kind == "builder"
    assert options.stage == "goid"
    assert options.version == "2.0.0"
    assert options.enabled_by_default is False
    assert options.severity == "soft_fail"
    assert len(options.inputs) == 1
    assert options.provides == ["capability"]
    assert options.requires == ["dependency"]
    assert options.depends_on == ["other.plugin"]
    assert options.resource_hints is hints
    assert options.requires_isolation is True
    assert options.isolation_kind == "process"
    assert options.tags == ["tag1", "tag2"]


# =============================================================================
# validate_option_keys Tests
# =============================================================================


def test_validate_option_keys_valid() -> None:
    """Verify validate_option_keys accepts valid keys."""
    allowed = {"name", "version", "description"}
    provided = {"name": "test", "version": "1.0"}

    # Should not raise
    BasePluginMetaOptions.validate_option_keys(allowed, provided)


def test_validate_option_keys_unknown_raises() -> None:
    """Verify validate_option_keys raises for unknown keys."""
    allowed = {"name", "version"}
    provided = {"name": "test", "unknown_key": "value"}

    with pytest.raises(ValueError, match="Unsupported plugin option keys"):
        BasePluginMetaOptions.validate_option_keys(allowed, provided)


def test_validate_option_keys_multiple_unknown() -> None:
    """Verify validate_option_keys reports all unknown keys."""
    allowed = {"name"}
    provided = {"name": "test", "bad1": "x", "bad2": "y"}

    with pytest.raises(ValueError, match=r"Unsupported plugin option keys") as exc_info:
        BasePluginMetaOptions.validate_option_keys(allowed, provided)

    error_msg = str(exc_info.value)
    assert "bad1" in error_msg
    assert "bad2" in error_msg


def test_validate_option_keys_empty_provided() -> None:
    """Verify validate_option_keys accepts empty provided dict."""
    allowed = {"name", "version"}

    # Should not raise
    BasePluginMetaOptions.validate_option_keys(allowed, {})


def test_validate_option_keys_empty_allowed() -> None:
    """Verify validate_option_keys rejects any keys if allowed is empty."""
    allowed: set[str] = set()
    provided = {"name": "test"}

    with pytest.raises(ValueError, match="Unsupported"):
        BasePluginMetaOptions.validate_option_keys(allowed, provided)


# =============================================================================
# to_base_metadata Tests
# =============================================================================


def test_to_base_metadata_uses_function_name() -> None:
    """Verify to_base_metadata uses function name when not provided."""
    options = BasePluginMetaOptions()

    def my_plugin_function(_ctx: object) -> PluginResult:
        return PluginResult.ok()

    metadata = options.to_base_metadata(my_plugin_function)

    assert metadata.name == "my.plugin.function"


def test_to_base_metadata_uses_custom_name() -> None:
    """Verify to_base_metadata uses provided name over function name."""
    options = BasePluginMetaOptions(name="custom.name")

    def my_function(_ctx: object) -> PluginResult:
        return PluginResult.ok()

    metadata = options.to_base_metadata(my_function)

    assert metadata.name == "custom.name"


def test_to_base_metadata_uses_function_docstring() -> None:
    """Verify to_base_metadata uses function docstring for description."""
    options = BasePluginMetaOptions()

    def documented_function(_ctx: object) -> PluginResult:
        """Execute the documented function.

        Returns
        -------
        PluginResult
            Successful plugin result.
        """
        return PluginResult.ok()

    metadata = options.to_base_metadata(documented_function)

    assert metadata.description == "Execute the documented function."


def test_to_base_metadata_uses_custom_description() -> None:
    """Verify to_base_metadata uses provided description over docstring."""
    options = BasePluginMetaOptions(description="Custom description")

    def documented_function(_ctx: object) -> PluginResult:
        """Execute and return success.

        Returns
        -------
        PluginResult
            Successful plugin result.
        """
        return PluginResult.ok()

    metadata = options.to_base_metadata(documented_function)

    assert metadata.description == "Custom description"


def test_to_base_metadata_strips_whitespace() -> None:
    """Verify to_base_metadata strips description whitespace."""
    options = BasePluginMetaOptions()

    def func(_ctx: object) -> PluginResult:
        """Handle description with whitespace.

        Returns
        -------
        PluginResult
            Successful plugin result.
        """
        return PluginResult.ok()

    metadata = options.to_base_metadata(func)

    assert metadata.description == "Handle description with whitespace."


def test_to_base_metadata_empty_docstring() -> None:
    """Verify to_base_metadata handles missing docstring."""
    options = BasePluginMetaOptions()

    def no_doc_function(_ctx: object) -> PluginResult:
        return PluginResult.ok()

    metadata = options.to_base_metadata(no_doc_function)

    assert not metadata.description


def test_to_base_metadata_uses_default_kind() -> None:
    """Verify to_base_metadata uses default kind when not specified."""
    options = BasePluginMetaOptions()

    def func(_ctx: object) -> PluginResult:
        return PluginResult.ok()

    metadata = options.to_base_metadata(func, default_kind="builder")

    assert metadata.kind == "builder"


def test_to_base_metadata_uses_default_stage() -> None:
    """Verify to_base_metadata uses default stage when not specified."""
    options = BasePluginMetaOptions()

    def func(_ctx: object) -> PluginResult:
        return PluginResult.ok()

    metadata = options.to_base_metadata(func, default_stage="graph")

    assert metadata.stage == "graph"


def test_to_base_metadata_overrides_defaults() -> None:
    """Verify to_base_metadata uses options over defaults."""
    options = BasePluginMetaOptions(kind="metric", stage="function")

    def func(_ctx: object) -> PluginResult:
        return PluginResult.ok()

    metadata = options.to_base_metadata(
        func,
        default_kind="builder",
        default_stage="other",
    )

    assert metadata.kind == "metric"
    assert metadata.stage == "function"


def test_to_base_metadata_converts_sequences() -> None:
    """Verify to_base_metadata converts sequences to tuples."""
    options = BasePluginMetaOptions(
        provides=["cap1", "cap2"],
        requires=["req1"],
        depends_on=["dep1", "dep2", "dep3"],
        tags=["tag1"],
    )

    def func(_ctx: object) -> PluginResult:
        return PluginResult.ok()

    metadata = options.to_base_metadata(func)

    assert isinstance(metadata.provides, tuple)
    assert metadata.provides == ("cap1", "cap2")
    assert isinstance(metadata.requires, tuple)
    assert metadata.requires == ("req1",)
    assert isinstance(metadata.depends_on, tuple)
    assert metadata.depends_on == ("dep1", "dep2", "dep3")
    assert isinstance(metadata.tags, tuple)
    assert metadata.tags == ("tag1",)


def test_to_base_metadata_preserves_all_fields() -> None:
    """Verify to_base_metadata preserves all option fields."""
    hints = PluginResourceHints(max_runtime_ms=1000)
    input_spec = PluginInputSpec(name="config", type_ref="Config")

    options = BasePluginMetaOptions(
        name="test.plugin",
        description="Test plugin",
        kind="analytics",
        stage="function",
        version="3.0.0",
        enabled_by_default=False,
        severity="skip_on_error",
        inputs=[input_spec],
        provides=["cap"],
        requires=["req"],
        depends_on=["dep"],
        resource_hints=hints,
        requires_isolation=True,
        isolation_kind="thread",
        tags=["test"],
    )

    def func(_ctx: object) -> PluginResult:
        return PluginResult.ok()

    metadata = options.to_base_metadata(func)

    assert metadata.name == "test.plugin"
    assert metadata.description == "Test plugin"
    assert metadata.kind == "analytics"
    assert metadata.stage == "function"
    assert metadata.version == "3.0.0"
    assert metadata.enabled_by_default is False
    assert metadata.severity == "skip_on_error"
    assert len(metadata.inputs) == 1
    assert metadata.provides == ("cap",)
    assert metadata.requires == ("req",)
    assert metadata.depends_on == ("dep",)
    assert metadata.resource_hints is hints
    assert metadata.requires_isolation is True
    assert metadata.isolation_kind == "thread"
    assert metadata.tags == ("test",)


# =============================================================================
# BasePluginMetaOptionsInput Tests
# =============================================================================


def test_base_plugin_meta_options_input_type() -> None:
    """Verify BasePluginMetaOptionsInput is a valid TypedDict."""
    # TypedDict with total=False allows partial dicts
    options: BasePluginMetaOptionsInput = {
        "name": "test",
        "kind": "analytics",
    }

    assert options["name"] == "test"
    assert options["kind"] == "analytics"


def test_base_plugin_meta_options_input_all_fields() -> None:
    """Verify BasePluginMetaOptionsInput accepts all fields."""
    options: BasePluginMetaOptionsInput = {
        "name": "test",
        "description": "desc",
        "kind": "builder",
        "stage": "goid",
        "version": "1.0",
        "enabled_by_default": True,
        "severity": "fatal",
        "inputs": [],
        "outputs": [],
        "provides": ["cap"],
        "requires": ["req"],
        "depends_on": ["dep"],
        "resource_hints": None,
        "requires_isolation": False,
        "isolation_kind": "none",
        "tags": ["tag"],
    }

    assert options["name"] == "test"
    assert options["tags"] == ["tag"]
